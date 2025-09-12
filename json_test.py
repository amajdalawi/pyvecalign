
import json
# src_en = []
# with open('some_file.json','r') as f:
#     srcen = json.load(f)

# with open('some_file2.json','r') as f:
#     srcfr = json.load(f)

# print(srcfr)

# print(srcen)

with open('nl_novel2.json','r') as f:
    book_dict_fr = json.load(f)



with open('en_novel2.json','r') as f:
    book_dict_en = json.load(f)

# print(book_dict_fr)
for key, val in book_dict_fr.items():
    print(f"Key: {key} val: {[x[:50] for x in val[:5]]}")

for key, val in book_dict_en.items():
    print(f"Key: {key} val: {[x[:50] for x in val[:5]]}")
