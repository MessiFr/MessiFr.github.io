import os
import re
import json
from tqdm import tqdm

def get_files():
    files = []
    for dirpath, dirnames, filenames in os.walk('public'):
        for filename in filenames:
            if filename.endswith('.md'):
                if not filename.endswith('homepage.md'):
                    files.append(os.path.join(dirpath, filename))

    return files

md_files = get_files()

def modify_md_file(md_file_path):
    # 读取文件内容
    with open(md_file_path, 'r') as f:
        content = f.read()

    # 使用正则表达式提取图片路径
    pattern = r'\!\[.*?\]\((.*?)\)'
    matches = re.findall(pattern, content)

    # 遍历所有匹配的图片路径
    for match in matches:
        # 如果路径不以/docs/开头，则替换为以/docs/开头的路径
        if not match.startswith('/docs/'):
            # 获取文件名和文件夹名
            filename = os.path.basename(match)
            dirname = os.path.dirname(match)
            # print(dirname)

            # 构造新的路径
            starts = '/' + '/'.join(md_file_path.split('/')[1:-1]) + '/'

            new_path = starts + os.path.join(dirname, filename)

            # 替换旧路径为新路径
            content = content.replace(match, new_path)
    

    dollar_pattern = r'(?<!\$)\$(?!\$)'
    # dollar_matches = re.findall(dollar_pattern, content)

    # for i in tqdm(range(len(dollar_matches))):
    #     match = dollar_matches[i]
    #     content = content.replace(match, '$$')
    content = re.sub(dollar_pattern, '$$', content)

    # 写入修改后的文件内容
    with open(md_file_path, 'w') as f:
        f.write(content)

for path in md_files:
    if '.md' in path:
        print(path)
        modify_md_file(path)

print("================== Rewrite .md Files Finished")

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
    doc_type = filename.split('.')[-1]
    new_item = {'name': name,
                'id': id,
                'desc': filename,
                'path': md_file_path,
                'title': title,
                'doc_type': doc_type}
    id += 1
    current_level.append(new_item)

# print(data)

# context = json.dumps(info, indent=2)

with open('doc_config.json', 'w') as f:
    json.dump(info, f, indent=2)

print("================== doc_config.json Finished")



new_data = data[0]['children']

order = ['notes', 'projects', 'development']

new_data = [item for item in new_data]
new_data = sorted(new_data, key=lambda x:order.index(x['desc']))

context = "const data = " + json.dumps(new_data, indent=2) + ";\nexport default data;"

with open('src/views/documents/info/doc.js', 'w') as f:
    f.write(context)

print("================== doc.js Finished")


def count_md_files(path):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.md'):
                count += 1
    return count


# 计算文件个数
doc_path = 'public/docs/'
project_path = 'public/docs/projects/'
digit_art = 'src/assets/img/digit_arts/'
doc_count = count_md_files(doc_path)


info_counts = {
    "documents" : doc_count,
    "projects" : len(os.listdir(project_path))-1,
    "digit_arts": len(os.listdir(digit_art))-1
}

context = "const doc_counts = " + json.dumps(info_counts, indent=2) + ";\nexport default doc_counts;"

with open('src/views/documents/info/count.js', 'w') as f:
    f.write(context)
print("================== count.js Finished")


# 定义图片文件夹路径
img_path = "src/assets/img"

# 定义正则表达式匹配文件名中的信息
pattern = re.compile(r"(.+)_@(.+)\.")

# 遍历digital_arts文件夹中的所有图片
digital_arts = []
digital_arts_path = os.path.join(img_path, "digit_arts")
for filename in os.listdir(digital_arts_path):
    if filename == ".DS_Store":
        continue
    # print(filename)
    # 匹配文件名中的信息
    match = pattern.match(filename)
    # print(match)
    if match:
        img_path = os.path.join("assets", "img", "digit_arts", filename)
        title = match.group(1)
        author = "@" + match.group(2)
        digital_arts.append({
            "img": f'require("{img_path}")',
            # "img": img_path,
            "title": title,
            "author": author,
            "type": "da"
        })
    else:
        img_path = os.path.join("assets", "img", "digit_arts", filename)
        title = filename.split('.')[0]
        digital_arts.append({
            "img": f'require("{img_path}")',
            # "img": img_path,
            "title": title,
            "type": "da"
        })

# 将digital_arts保存到digital.js文件中
content = "const DigitalArts = " + json.dumps(digital_arts, indent=2) + ";\nexport default DigitalArts;"
with open("src/views/documents/info/digital_arts.js", "w") as f:
    f.write(content)
    

# 遍历bg_collection文件夹中的所有图片

img_path = "src/assets/img"
gallery_bg = []
gallery_bg_path = os.path.join(img_path, "bg_collection")
for filename in os.listdir(gallery_bg_path):
    if filename == ".DS_Store":
        continue
    # 匹配文件名中的信息
    match = pattern.match(filename)
    if match:
        img_path = os.path.join("assets", "img", "bg_collection", filename)
        title = match.group(1)
        author = "@" + match.group(2) if match.group(2) else ""
        gallery_bg.append({
            "img": f'require("{img_path}")',
            # "img": img_path,
            "title": title,
            "author": author,
            "type": "bg"
        })
    else:
        img_path = os.path.join("assets", "img", "bg_collection", filename)
        title = filename.split('.')[0]
        gallery_bg.append({
            "img": f'require("{img_path}")',
            # "img": img_path,
            "title": title,
            "type": "bg"
        })

# 将gallery_bg保存到gallery_bg.js文件中
content = "const GalleryBg = " + json.dumps(gallery_bg, indent=2) + ";\nexport default GalleryBg;"
with open("src/views/documents/info/gallery_bg.js", "w") as f:
    f.write(content)
# print(gallery_bg)


src_pattern = r'\\\"(.*)\\\"'

# 读取文件并替换每一行中匹配到的字符串
with open('src/views/documents/info/digital_arts.js', 'r') as f:
    lines = f.readlines()

with open('src/views/documents/info/digital_arts.js', 'w') as f:
    for line in lines:
        src = re.findall(src_pattern, line)
        if src:
            f.write(f"    \"img\": require(\"{src[0]}\"),\n")
        else:
            f.write(line)

print("================== Gallery Digital Arts config Finished")


# 读取文件并替换每一行中匹配到的字符串
with open('src/views/documents/info/gallery_bg.js', 'r') as f:
    lines = f.readlines()

with open('src/views/documents/info/gallery_bg.js', 'w') as f:
    for line in lines:
        src = re.findall(src_pattern, line)
        if src:
            f.write(f"    \"img\": require(\"{src[0]}\"),\n")
        else:
            f.write(line)

print("================== Gallery Background config Finished")



print("==================")
print("Successful")
print("==================")