"""Этап 4 — LLM раскладывает подграф на REQUIRED / USEFUL / IRRELEVANT.

Что прокидываем модели:
* свободный пользовательский запрос;
* anchor — единственный класс, который изначально гарантированно релевантен;
* список всех узлов подграфа в усечённом виде (`_node_for_llm`);
* список рёбер подграфа в усечённом виде (`_edge_for_llm`).

Что НЕ прокидываем:
* поле `description` узла. Оно сгенерировано LLM на стадии подготовки
  данных и часто пересекается формулировками с эталоном — это утечка.
* полные сигнатуры (методов больше 30, параметров больше 20). Это и
  гигиена контекстного окна, и снижение поверхности утечки.

Контракт результата:
* `node_ids` — все классы, которые пайплайн в итоге оставляет в подграфе
  (т.е. `required ∪ useful`). Anchor дополнительно гарантируется в
  required, если LLM почему-то его не упомянул.
* `payload.required` / `payload.useful` — раздельные множества, нужны
  и для `to_diagram()`, и для метрик.
"""

from __future__ import annotations

import time
import traceback
from typing import Any

from src.approaches.anchor_neighbors import prompt_templates
from src.approaches.anchor_neighbors.llm_trace import LLMTracer
from src.approaches.anchor_neighbors.stage_outputs import StageName, StageOutcome
from src.llm.client import LLMClient
from src.llm.parser import parse_json_response


_METHODS_PER_NODE = 30
_PARAMS_PER_NODE = 20


def _short_name(node_id: str) -> str:
    return node_id.rsplit(".", 1)[-1] if "." in node_id else node_id


def _node_for_llm(node: dict[str, Any]) -> dict[str, Any]:
    """Узкий снимок узла для LLM — без `description`, с обрезанными
    методами/параметрами."""
    return {
        "node_id": node.get("node_id"),
        "name": node.get("name") or _short_name(node.get("node_id", "")),
        "type": node.get("type", "class"),
        "methods": (node.get("methods") or []),
        "params": (node.get("params") or []),
    }


def _edge_for_llm(edge: dict[str, Any]) -> dict[str, Any]:
    """Узкий снимок ребра — оставляем только направление и тип связи."""
    return {
        "from": edge.get("node_id_from"),
        "to": edge.get("node_id_to"),
        "kind": edge.get("description") or edge.get("kind") or "",
    }


async def _execute_single_prune_step(
    *,
    step_name: str,
    step_index: int,
    query: str,
    sub_nodes: list[dict[str, Any]],
    sub_edges: list[dict[str, Any]],
    context: dict[str, Any],
    llm: LLMClient,
    tracer: LLMTracer | None,
    sample_id: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Выполнить один шаг multi-step прунинга.
    
    Возвращает:
    - data: результат парсинга JSON (или None при ошибке)
    - info: метаданные вызова (timing, tokens, errors)
    """
    system_prompt = prompt_templates.prune_step_system(step_name)
    user_prompt = prompt_templates.prune_step_user(
        step_name=step_name,
        query=query,
        nodes=sub_nodes,
        edges=sub_edges,
        context=context,
    )
    
    stage_name_for_tracer = f"{StageName.PRUNE.value}_step{step_index + 1}_{step_name}"
    
    if tracer is not None and sample_id:
        tracer.record_request(StageName.PRUNE, f"{sample_id}_{stage_name_for_tracer}", 
                             system_prompt, user_prompt)
    
    started = time.time()
    try:
        resp = await llm.call(system_prompt, user_prompt, json_mode=True)
    except Exception as e:  # noqa: BLE001
        tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        error_msg = (
            f"[stage4_prune, step {step_index + 1}: {step_name}] "
            f"LLM call failed for sample '{sample_id}'\n"
            f"Error type: {type(e).__name__}\n"
            f"Error message: {str(e)}\n"
            f"Traceback:\n{tb_str}"
        )
        if tracer is not None and sample_id:
            tracer.record_error(StageName.PRUNE, f"{sample_id}_{stage_name_for_tracer}", 
                              error_msg)
        info = {
            "step": step_index + 1,
            "step_name": step_name,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": tb_str,
            "elapsed_s": round(time.time() - started, 2),
        }
        return None, info
    
    if tracer is not None and sample_id:
        tracer.record_response(StageName.PRUNE, f"{sample_id}_{stage_name_for_tracer}", 
                              resp.content)
    
    info = {
        "step": step_index + 1,
        "step_name": step_name,
        "elapsed_s": round(time.time() - started, 2),
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
    }
    
    try:
        data = parse_json_response(resp.content)
    except ValueError as e:
        error_msg = (
            f"[stage4_prune, step {step_index + 1}: {step_name}] "
            f"JSON parsing failed for sample '{sample_id}'\n"
            f"Error: {str(e)}\n"
            f"Response excerpt: {resp.content[:200]}"
        )
        info["error"] = str(e)
        info["raw_excerpt"] = resp.content[:200]
        info["full_response"] = resp.content
        return None, info
    
    if not isinstance(data, dict):
        info["error"] = "Response is not a JSON object"
        info["raw_excerpt"] = resp.content[:200]
        return None, info
    
    return data, info


async def prune_subgraph(
    *,
    query: str,
    sub_nodes: list[dict[str, Any]],
    sub_edges: list[dict[str, Any]],
    llm: LLMClient,
    prune_steps: list[str] = None,
    tracer: LLMTracer | None = None,
    sample_id: str = "",
) -> StageOutcome:
    """Прогнать multi-step LLM-прунинг и вернуть структурированный StageOutcome.
    
    Общая логика без привязки к конкретным полям:
    - Каждый шаг возвращает произвольный JSON
    - Весь JSON автоматически попадает в context для следующего шага
    - Только ПОСЛЕДНИЙ шаг должен вернуть required/useful (это контракт stage 4)
    - Все промежуточные поля определяются промптами, не кодом
    
    Args:
        prune_steps: список имён шагов (например, ["identify_core", "classify_neighbors"]).
                     Если None или ["single"], выполняется одношаговый прунинг.
        tracer: записывает request/response для каждого шага.
        sample_id: идентификатор сэмпла для логирования.
    """
    if prune_steps is None or prune_steps == ["single"]:
        prune_steps = ["single"]
    
    started_total = time.time()
    valid_ids = {n["node_id"] for n in sub_nodes if n.get("node_id")}
    
    # Контекст, передаваемый между шагами - накапливаем ВСЕ результаты
    context: dict[str, Any] = {}
    # Метаданные всех шагов
    steps_info: list[dict[str, Any]] = []
    
    # Результат последнего шага
    last_step_data: dict[str, Any] | None = None
    
    # Выполняем шаги последовательно
    for step_idx, step_name in enumerate(prune_steps):
        data, step_info = await _execute_single_prune_step(
            step_name=step_name,
            step_index=step_idx,
            query=query,
            sub_nodes=sub_nodes,
            sub_edges=sub_edges,
            context=context,
            llm=llm,
            tracer=tracer,
            sample_id=sample_id,
        )
        
        # Сохраняем ВЕСЬ response в метаданные (для диагностики)
        step_info["response_data"] = data if data is not None else None
        steps_info.append(step_info)
        
        # Если ошибка — прерываем
        if data is None:
            return StageOutcome(
                stage=StageName.PRUNE,
                aborted=f"step{step_idx + 1}_failed",
                info={
                    "subgraph_input_size": len(sub_nodes),
                    "total_steps": len(prune_steps),
                    "failed_at_step": step_idx + 1,
                    "steps": steps_info,
                    "elapsed_total_s": round(time.time() - started_total, 2),
                },
            )
        
        # ОБЩАЯ ЛОГИКА: весь JSON response идёт в context для следующих шагов
        # Никаких проверок на конкретные поля!
        for key, value in data.items():
            context[key] = value
        
        # Запоминаем результат последнего шага
        last_step_data = data
    
    # КОНТРАКТ: последний шаг ОБЯЗАН вернуть required/useful
    if last_step_data is None:
        return StageOutcome(
            stage=StageName.PRUNE,
            aborted="no_steps_executed",
            info={
                "subgraph_input_size": len(sub_nodes),
                "total_steps": len(prune_steps),
                "steps": steps_info,
                "elapsed_total_s": round(time.time() - started_total, 2),
            },
        )
    
    # Извлекаем required/useful из последнего шага
    # Только эти поля обязательны - всё остальное определяется промптами
    required = {
        x for x in (last_step_data.get("required") or [])
        if isinstance(x, str) and x in valid_ids
    }
    useful = {
        x for x in (last_step_data.get("useful") or [])
        if isinstance(x, str) and x in valid_ids and x not in required
    }
    
    # Итоговая информация
    info = {
        "subgraph_input_size": len(sub_nodes),
        "total_steps": len(prune_steps),
        "steps": steps_info,
        "elapsed_total_s": round(time.time() - started_total, 2),
    }
    
    keep = required | useful
    return StageOutcome(
        stage=StageName.PRUNE,
        node_ids=sorted(keep),
        payload={
            "required": sorted(required),
            "useful": sorted(useful),
        },
        info=info,
    )


# Обратная совместимость: старая функция для одношагового прунинга
async def prune_subgraph_legacy(
    *,
    query: str,
    sub_nodes: list[dict[str, Any]],
    sub_edges: list[dict[str, Any]],
    llm: LLMClient,
    tracer: LLMTracer | None = None,
    sample_id: str = "",
) -> StageOutcome:
    """Legacy версия для обратной совместимости.
    
    Использует оригинальные промпты prune_system.txt / prune_user.txt.
    Новый код должен использовать prune_subgraph() с prune_steps.
    """
    system_prompt = prompt_templates.prune_system()
    user_prompt = prompt_templates.prune_user(
        query=query,
        nodes=sub_nodes,
        edges=sub_edges,
    )

    if tracer is not None and sample_id:
        tracer.record_request(StageName.PRUNE, sample_id, system_prompt, user_prompt)

    started = time.time()
    try:
        resp = await llm.call(system_prompt, user_prompt, json_mode=True)
    except Exception as e:  # noqa: BLE001 — общая точка обработки внешних сбоев
        # Получаем полный traceback для детальной диагностики
        tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        error_msg = (
            f"[stage4_prune] LLM call failed for sample '{sample_id}'\n"
            f"Error type: {type(e).__name__}\n"
            f"Error message: {str(e)}\n"
            f"Traceback:\n{tb_str}"
        )
        if tracer is not None and sample_id:
            tracer.record_error(StageName.PRUNE, sample_id, error_msg)
        return StageOutcome(
            stage=StageName.PRUNE,
            aborted="llm_call_failed",
            info={
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": tb_str,
                "elapsed_s": round(time.time() - started, 2),
            },
        )

    if tracer is not None and sample_id:
        tracer.record_response(StageName.PRUNE, sample_id, resp.content)

    info = {
        "elapsed_s": round(time.time() - started, 2),
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "subgraph_input_size": len(sub_nodes),
    }

    try:
        data = parse_json_response(resp.content)
    except ValueError as e:
        error_msg = (
            f"[stage4_prune] JSON parsing failed for sample '{sample_id}'\n"
            f"Error: {str(e)}\n"
            f"Response excerpt: {resp.content[:200]}"
        )
        return StageOutcome(
            stage=StageName.PRUNE,
            aborted="bad_json",
            info={
                **info,
                "error": str(e),
                "raw_excerpt": resp.content[:200],
                "full_response": resp.content,
            },
        )
    if not isinstance(data, dict):
        return StageOutcome(
            stage=StageName.PRUNE,
            aborted="bad_json_shape",
            info={**info, "raw_excerpt": resp.content[:200]},
        )

    # Доверяем только тем node_id, которые реально были в подграфе.
    valid_ids = {n["node_id"] for n in sub_nodes if n.get("node_id")}
    required = {
        x for x in (data.get("required") or [])
        if isinstance(x, str) and x in valid_ids
    }
    useful = {
        x for x in (data.get("useful") or [])
        if isinstance(x, str) and x in valid_ids and x not in required
    }

    # Сохраняем reasoning если есть (для диагностики)
    reasoning = data.get("reasoning", "")
    if reasoning:
        info["reasoning"] = reasoning

    # Anchor по определению релевантен — гарантируем его наличие.

    keep = required | useful
    return StageOutcome(
        stage=StageName.PRUNE,
        node_ids=sorted(keep),
        payload={
            "required": sorted(required),
            "useful": sorted(useful),
        },
        info=info,
    )
