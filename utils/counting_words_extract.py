from typing import Dict
import re

start_token = "<|startoftext|>"
end_token = "<|endoftext|>"

pattern = re.compile(r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\s(\w+)')


word2number = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                  'six': 6, 'seven':7, 'eight': 8, 'nine': 9, 'ten': 10
                  }

def find_nummod(sentences):
    results = []
    for sentence in sentences:
        match = pattern.search(sentence)
        if match:
            # Extract the number word and the object noun
            nummod = [match.group(1), match.group(2)]
            results.append(nummod)
    return results


def extract_attribution_indices_nummod(prompt, parser):
    doc = parser(prompt)
    subtrees = []
    modifiers = ["nummod"]
    for w in doc:
        if w.pos_ not in ["NOUN", "PROPN"] or w.dep_ in modifiers:
            continue
        subtree = []
        stack = []
        for child in w.children:
            if child.dep_ in modifiers:
                subtree.append(child)
                stack.extend(child.children)

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == "conj":
                subtree.append(node)
                stack.extend(node.children)
        if subtree:
            subtree.append(w)
            subtrees.append(subtree)
    return subtrees

def get_indices(tokenizer, prompt: str) -> Dict[str, int]:
    """Utility function to list the indices of the tokens you wish to alter"""
    ids = tokenizer(prompt).input_ids
    indices = {
        i: tok
        for tok, i in zip(
            tokenizer.convert_ids_to_tokens(ids), range(len(ids))
        )
    }
    return indices

def align_wordpieces_indices(
        wordpieces2indices, start_idx, target_word
):
    """
    Aligns a `target_word` that contains more than one wordpiece (the first wordpiece is `start_idx`)
    """

    wp_indices = [start_idx]
    wp = wordpieces2indices[start_idx].replace("</w>", "")

    # Run over the next wordpieces in the sequence (which is why we use +1)
    for wp_idx in range(start_idx + 1, len(wordpieces2indices)):
        if wp == target_word:
            break

        wp2 = wordpieces2indices[wp_idx].replace("</w>", "")
        if target_word.startswith(wp + wp2) and wp2 != target_word:
            wp += wordpieces2indices[wp_idx].replace("</w>", "")
            wp_indices.append(wp_idx)
        else:
            wp_indices = (
                []
            )  # if there's no match, you want to clear the list and finish
            break

    return wp_indices