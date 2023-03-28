import os
import re
import json

def get_files():
    files = []
    for dirpath, dirnames, filenames in os.walk('public'):
        for filename in filenames:
            if filename.endswith('.md') or filename.endswith('.pdf'):
                if not filename.endswith('homepage.md'):
                    files.append(os.path.join(dirpath, filename))

    return files

md_files = get_files()

def modify_md_file(md_file_path):
    # 读取文件内容
    with open(md_file_path, 'r') as f:
        content = f.read()

    # 使用正则表达式提取图片路径
    pattern = r'!\[.*?\]\((.*?)\)'
    matches = re.findall(pattern, content)

    # 遍历所有匹配的图片路径
    for match in matches:
        # 如果路径不以/docs/开头，则替换为以/docs/开头的路径
        if not match.startswith('/docs/'):
            # 获取文件名和文件夹名
            filename = os.path.basename(match)
            dirname = os.path.dirname(match)
            print(dirname)

            # 构造新的路径
            starts = '/' + '/'.join(md_file_path.split('/')[1:]) + '/'
            new_path = starts + os.path.join(dirname, filename)

            # 替换旧路径为新路径
            content = content.replace(match, new_path)

    # 写入修改后的文件内容
    with open(md_file_path, 'w') as f:
        f.write(content)

for path in md_files:
    if '.md' in path:
        modify_md_file(path)


with open('doc_config.json') as f:
    info = json.load(f)

def get_info(desc):
    for item in info:
        if desc in item:
            return item[desc]['name'], item[desc]['title']
    info.append(
        {
            desc: {
                'name': desc,
                'title': desc
            }
        }
    )
    return desc, desc

data = []

docs_dir = 'public/docs/'

id = 0

for md_file in md_files:
    # remove the 'public/' prefix and split the path
    md_file_path = md_file[len('public/'):]
    md_file_parts = md_file_path.split('/')

    # initialize variables for traversing the data structure
    current_level = data
    current_path = docs_dir

    # traverse the data structure to find the correct location to add the new item
    for md_file_part in md_file_parts[:-1]:
        current_path = os.path.join(current_path, md_file_part)
        found_item = None
        for item in current_level:
            if item.get('desc') == md_file_part:
                found_item = item
                break
        if found_item is None:
            name, title = get_info(md_file_part)
            new_item = {'name': name,
                        'id': id,
                        'desc': md_file_part,
                        'children': []}
            id += 1
            current_level.append(new_item)
            current_level = new_item['children']
        else:
            current_level = found_item['children']

    # add the new item to the data structure
    filename = md_file_parts[-1]
    name, title = get_info(filename)
    new_item = {'name': name,
                'id': id,
                'desc': filename,
                'path': md_file_path,
                'title': title}
    id += 1
    current_level.append(new_item)

# print(data)

# context = json.dumps(info, indent=2)

with open('doc_config.json', 'w') as f:
    json.dump(info, f, indent=2)


new_data = data[0]['children']

order = ['notes', 'projects', 'development']

new_data = [item for item in new_data]
new_data = sorted(new_data, key=lambda x:order.index(x['desc']))

context = "const data = " + json.dumps(new_data, indent=2) + ";\nexport default data;"

with open('src/views/documents/plugin/doc.js', 'w') as f:
    f.write(context)

print("***")
print("Done")
print("***")