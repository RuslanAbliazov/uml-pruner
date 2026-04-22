import json
import os
import argparse

def remove_description_from_json(input_path, output_path):
    """
    Удаляет поле 'description' из всех объектов в массиве 'nodes' JSON-файла.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'nodes' in data and isinstance(data['nodes'], list):
        for node in data['nodes']:
            if 'description' in node:
                del node['description']
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description='Remove description fields from JSON diagrams')
    parser.add_argument('--input_dir', default='full_diagrams_fixed_generic', help='Input directory with JSON files')
    parser.add_argument('--output_dir', default='cleaned_diagrams', help='Output directory for processed files')
    
    args = parser.parse_args()
    
    # Создаем выходную директорию если нужно
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Обрабатываем каждый JSON-файл в директории
    for filename in os.listdir(args.input_dir):
        if filename.endswith('.json'):
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, filename)
            
            print(f"Processing: {filename}")
            remove_description_from_json(input_path, output_path)
    
    print(f"Completed! Processed files saved to: {args.output_dir}")

if __name__ == '__main__':
    main()