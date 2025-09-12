from pathlib import Path
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
from pprint import pprint as pp

from bs4.element import Tag

def print_body_tree(soup, indent_unit="---- ", include_body=False, skip_tags=None):
    """
    Print the element tree of the document <body> (HTML or XHTML) using ASCII indentation.

    Parameters
    ----------
    soup : bs4.BeautifulSoup
        Parsed document.
    indent_unit : str
        The string repeated per depth level (default: "---- ").
    include_body : bool
        If True, include the <body> node itself at the top.
    skip_tags : set[str] | None
        Tag names to skip entirely (default: {"script","style","noscript","template"}).
    """
    if skip_tags is None:
        skip_tags = {"script", "style", "noscript", "template"}

    body = soup.body or soup.find("body") or soup  # fallback if no <body>

    def walk(node: Tag, depth: int):
        for child in node.children:
            if isinstance(child, Tag):
                if child.name in skip_tags:
                    continue
                print(f"{indent_unit * depth}{child.name}")
                walk(child, depth + 1)

    if include_body and isinstance(body, Tag) and body.name != "[document]":
        print(body.name)
        walk(body, 1)
    else:
        walk(body, 0)


# from bs4 import BeautifulSoup

def print_dom_tree(element, indent=0, preview_chars=0):
    """
    Recursively print the DOM tree of a BeautifulSoup element.

    Args:
        element: The BeautifulSoup element to print.
        indent: Current indentation level.
        preview_chars: If > 0, show up to this many characters of text content.
    """
    for child in element.children:
        if child.name:  # It's a tag
            line = "----" * indent + child.name
            if preview_chars > 0:
                text = child.get_text(strip=True)
                if text:
                    snippet = text[:preview_chars].replace("\n", " ")
                    line += f"  [{snippet}{'...' if len(text) > preview_chars else ''}]"
            print(line)
            print_dom_tree(child, indent + 1, preview_chars)

BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "div", "dl", "fieldset", "figcaption",
    "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6",
    "header", "li", "main", "nav", "ol", "p", "pre", "section", "table", "ul"
}

def lowest_block_texts(soup):
    """
    Traverse the DOM and return text content of the lowest-level block elements.
    Inline children (span, i, etc.) are ignored in favor of their block parent.
    """
    results = []

    for tag in soup.body.find_all(BLOCK_TAGS):
        # If this block tag contains *other block tags*, skip it
        if tag.find(BLOCK_TAGS):
            continue
        # Otherwise, it's the lowest block in this branch → collect text
        text = tag.get_text(strip=True)
        if text:
            results.append(text)

    return results


ebookfr = Path('./joefr.epub')
book = epub.read_epub(ebookfr)
i = 0
for item in book.get_items():
    if item.get_type() == ebooklib.ITEM_DOCUMENT:
        soup = BeautifulSoup(item.get_content(), features="html.parser")
        print_dom_tree(soup,preview_chars=20)
        i = i + 1
        if i > 10:
          break


from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
from pathlib import Path
from pprint import pprint as pp
import re


ebook_path = Path('./nl.epub')
# ebook_path = Path('Abercrombie, Joe - De Eerste Wet 1 - Macht van het zwaard.epub')
book = epub.read_epub(ebook_path)

# def print_body_tree(soup):

with open("some_file2.txt","w") as f:
  for item in book.get_items():
      if item.get_type() == ebooklib.ITEM_DOCUMENT:
          # print(item.get_id())
          print('############################')
          print(item.get_name())
          f.write(item.get_name())
          f.write('\n')
          # print(item.get_type())
          soup = BeautifulSoup(item.get_content(), features="html.parser")
          # print_body_tree(soup)
          l = lowest_block_texts(soup)
          # pp(l)
          for i in range(len(l)):
              print(f'{i+1}. {l[i][:30]}')
              f.write(f'{i+1}. {l[i][:50]}\n')

          print('############################')

# ebook_path  = Path('./joeen.epub')
# book = epub.read_epub(ebook_path)

# with open("some_file.txt","a") as f:
#   for item in book.get_items():
#       if item.get_type() == ebooklib.ITEM_DOCUMENT:
#           # print(item.get_id())
#           print('############################')
#           print(item.get_name())
#           f.write(item.get_name())
#           # print(item.get_type())
#           soup = BeautifulSoup(item.get_content(), features="html.parser")
#           # print_body_tree(soup)
#           l = lowest_block_texts(soup)
#           # pp(l)
#           for i in range(len(l)):
#               print(f'{i+1}. {l[i][:30]}')
#               f.write(f'{i+1}. {l[i][:30]}\n')

#           print('############################')
book= epub.read_epub("pt.epub")
import json
# s2 = []
book_dict = {}
for item in book.get_items():
    if item.get_type() == ebooklib.ITEM_DOCUMENT:
        # print(item.get_name())
        soup = BeautifulSoup(item.get_content(), features="html.parser")
        # print_body_tree(soup)
        l = lowest_block_texts(soup)
        book_dict[item.get_name()] = l
        # pp(l)
        # pp(l)
        # for el in l:
        #   s2.append(el)
        # for i in range(len(l)):
        #     print(f'{i+1}. {l[i][:30]}')

with open('pt_novel.json','w') as f:
  json.dump(book_dict,f)

# book= epub.read_epub("fr.epub")
# # s2 = []
# book_dict = {}
# for item in book.get_items():
#     if item.get_type() == ebooklib.ITEM_DOCUMENT:
#         # print(item.get_name())
#         soup = BeautifulSoup(item.get_content(), features="html.parser")
#         # print_body_tree(soup)
#         l = lowest_block_texts(soup)
#         book_dict[item.get_name()] = l
#         # pp(l)
#         pp(l)
# with open('fr_novel2.json','w') as f:
#   json.dump(book_dict,f)