import json
import os
import re
import regex
import sys
from copy import deepcopy
from glob import glob
from typing import Dict, List, Tuple

from sentsplit.config import en_config
from sentsplit.segment import SentSplit

from icecream import ic
import html
from pylatexenc.latexencode import unicode_to_latex
from pylatexenc.latex2text import LatexNodes2Text
import difflib
import pandas as pd
from datasets import load_dataset

PUNCTUATIONS = set(['.', '?', '!', ';', ':'])


class DirectLatexdiffParser:
    def __init__(self):
        self.anchor_symbol = 'ൠ'
        my_config = deepcopy(en_config)
        my_config['mincut'] = 10
        self.splitter = SentSplit('en', **my_config)

    def _write_to_latex(self, temp_path, source_abs, preprint_v1):
        with open(f'{temp_path}/{preprint_v1}.tex', 'w') as f:
            f.write('\\documentclass{article}\n')
            f.write('\\begin{document}\n')
            f.write('\\begin{abstract}\n')
            f.write(unicode_to_latex(source_abs))
            f.write('\\end{abstract}\n')
            f.write('\\end{document}\n')
        return f'{temp_path}/{preprint_v1}.tex'

    def _generate_latex_file(self, before_revision, after_revision, temp_path):
        if not os.path.isdir(temp_path):
            os.mkdir(temp_path)
        source_abs = before_revision.replace('%', '\%')
        target_abs = after_revision.replace('%', '\%')
        source_file = self._write_to_latex(temp_path, source_abs, 'source')
        target_file = self._write_to_latex(temp_path, target_abs, 'target')
        latexdiff_command_tex = "latexdiff "
        latexdiff_command_tex += "--ignore-warnings "
        latexdiff_command_tex += "--math-markup=0 "
        latexdiff_command_tex += source_file + " " + target_file + " > " + f'{temp_path}/source_diff_target.tex'
        os.system(latexdiff_command_tex)
        # delete non diff files
        #os.remove(f'{temp_path}/source.tex')
        #os.remove(f'{temp_path}/target.tex')
        return f'{temp_path}/source_diff_target.tex'

    def parse_latex_diff(self, before_revision, after_revision, temp_path) -> Dict:
        latex_diff_path = self._generate_latex_file(before_revision, after_revision, temp_path)
        with open(latex_diff_path) as f:
            raw_text = f.read()
        
        # extract abstract
        rgx_abstract = r'\\begin\{abstract\}(.+)\\end\{abstract\}'
        abstract_match = re.search(rgx_abstract, raw_text, re.DOTALL)
        if not abstract_match:
            print(f'ERROR: abstract not extracted for {latex_diff_path}')
            return
        raw_text = abstract_match.group(1)

        # drop paragraphs where there is no latexdiff commands
        raw_text = self.drop_irrelevant_parts(raw_text)

        # strip latex commands like \emph{..} etc.
        raw_text = self.strip_latex_command(raw_text)

        # standardize latexdiff commands
        raw_text = self.standardize_latexdiff_commands(raw_text)

        # remove space between punctuations when it is followed by latexdiff commands
        raw_text = self.adjust_punctuations(raw_text)

        before_revision, after_revision, edit_actions = self.parse_abstract(raw_text)
        if not before_revision:
            print('ERROR: sanity checks are not passed')
            return


        # add sentence segmentation info.
        # print(f'before_revision: {before_revision}')
        # input(abstract)
        before_sents = self.splitter.segment(before_revision)
        sents_positions = []
        cuml_char_index = 0
        for sent in before_sents:
            sents_positions.append(cuml_char_index)
            for _ in range(len(sent)):
                cuml_char_index += 1

        parsed_sample = {
            'before_revision': before_revision,
            'after_revision': after_revision,
            'edit_actions': edit_actions,
            'sents_char_pos': sents_positions
        }
        return parsed_sample

    def drop_irrelevant_parts(self, text: str) -> str:
        """Drop paragraphs that do not contain latexdiff commands"""
        filtered_paragraphs = []
        delbegin_count = addbegin_count = 0
        paragraphs = text.split('\n\n')
        excludes = set()
        for i, para in enumerate(paragraphs):
            if all(command not in para for command in ['\DIFdelbegin', '\DIFaddbegin', '\DIFdelend', '\DIFaddend']):
                # need to check the opened commands as a command may span across paragraphs
                if delbegin_count <= 0 and addbegin_count <= 0:
                    excludes.add(i)
            else:
                delbegin_count += para.count('\DIFdelbegin')
                addbegin_count += para.count('\DIFaddbegin')
                delbegin_count -= para.count('\DIFdelend')
                addbegin_count -= para.count('\DIFaddend')

        filtered_paragraphs = [para for i, para in enumerate(paragraphs) if i not in excludes]
        return '\n\n'.join(filtered_paragraphs)

    def strip_latex_command(self, text: str) -> str:
        """Strip latex commands like `\emph{content}`"""
        def _strip(matched: re.Match) -> str:
            excludes = set(['DIFdel', 'DIFadd'])
            command_name = matched.group(1)
            content = matched.group(2)
            if command_name in excludes:
                return matched.group(0)
            return content

        rgx_command = r'\\([\w\d]+)\{(.*?)\}'
        stripped_text = re.sub(rgx_command, _strip, text, re.DOTALL)
        return stripped_text

    def standardize_latexdiff_commands(self, text: str) -> str:
        """Remove unwanted giberish that is sometimes present in between latexdiff commands like, `%DIFDELCMD <%DIFDELCMD < %%%`
        and strip the outer latexdiff commands like DIFdel(begin|end) and DIFadd(begin|end)"""
        rgx_delbegin = r'\\DIFdelbegin\s*%DIFDELCMD < } %%%\s*\\DIFdelend'
        rgx_addbegin = r'\\DIFaddbegin\s*%DIFDELCMD < } %%%\s*\\DIFaddend'
        text = re.sub(rgx_delbegin, r'', text, re.DOTALL)
        text = re.sub(rgx_addbegin, r'', text, re.DOTALL)

        rgx_delbegin = r'(\\DIFdelbegin)[^\\]+?(?=\\DIFdel)'
        rgx_addbegin = r'(\\DIFaddbegin)[^\\]+?(?=\\DIFadd)'
        text = re.sub(rgx_delbegin, r'\1 ', text, re.DOTALL)
        text = re.sub(rgx_addbegin, r'\1 ', text, re.DOTALL)

        rgx_delend = r'(\\DIFdel\{(?:\\}|[^\}])*\})([^\\]+)(\\DIFdelend)'
        rgx_addend = r'(\\DIFadd\{(?:\\}|[^\}])*\})([^\\]+)(\\DIFaddend)'
        text = re.sub(rgx_delend, r'\1 \3', text, re.DOTALL)
        text = re.sub(rgx_addend, r'\1 \3', text, re.DOTALL)

        rgx_delend = r'(\\DIFdel\{(?:\\}|[^\}])*\})([^\\]+)(\\DIFdel)'
        rgx_addend = r'(\\DIFadd\{(?:\\}|[^\}])*\})([^\\]+)(\\DIFadd)'
        text = re.sub(rgx_delend, r'\1 \3', text, re.DOTALL)
        text = re.sub(rgx_addend, r'\1 \3', text, re.DOTALL)

        # Strip outer latexdiff commands
        text = text.replace('\DIFdelbegin', '')
        text = text.replace('\DIFdelend', '')
        text = text.replace('\DIFaddbegin', '')
        text = text.replace('\DIFaddend', '')
        return text

    def adjust_punctuations(self, text: str) -> str:
        """Remove any space before a punctuation that is present alone"""

        # print(f'before: {text}')
        punct_str = f"[{''.join(PUNCTUATIONS)}]"
        rgx_punct = rf'(\\DIF(del|add)end) +(' + punct_str + r')'
        rgx_punct = r"(\\DIF(del|add)\{[^\}]*\}) +([,.?!:;])"
        ## print(f'rgx_punct: {rgx_punct}')
        text = re.sub(rgx_punct, r'\1\3', text)
        ## print(f'after: {text}')
        rgx_punct = r"(\\ {1,}(\\\\DIF(del|add)\\{[,\\.\\?!:;]))"
        # print(re.sub(rgx_punct, r'\2', text))
        return re.sub(rgx_punct, r'\2', text)
        #return re.sub(rgx_punct, r'\1\3', text)

    def parse_abstract(self, abstract: str) -> Tuple[str, str, Dict]:
        """Parse a cleaned raw abstract to produce `before_revision`, `after_revision`, and `edit_actions`"""
        def _parse_abstract(matched: re.Match, action_type: str) -> str:
            """Process a matched regex by stripping the latexdiff commands, anchoring the contents, and defining an `edit_action`"""
            anchor = f'{self.anchor_symbol}{action_type}{self.anchor_symbol}'
            start_position = matched.start()

            if action_type == 'R':
                before = ' '.join(matched.group(1).strip().split())
                after = ' '.join(matched.group(2).strip().split())
            elif action_type == 'D':
                before = ' '.join(matched.group(1).strip().split())
                after = None
            elif action_type == 'A':
                before = None
                after = ' '.join(matched.group(1).strip().split())
            else:
                print(f'ERROR: unknown action_type, {action_type}')
                sys.exit()

            action = {
                'type': action_type,
                'before': before,
                'after': after
            }
            if action_type == 'R':
                replaced_string = f'{anchor}{before}'
                replace_actions.append((start_position, action))
            elif action_type == 'D':
                replaced_string = f'{anchor}{before}'
                delete_actions.append((start_position, action))
            else:
                replaced_string = f'{anchor}'
                add_actions.append((start_position, action))
            return replaced_string

        # replace action is when one delbegin is immediately followed by one addbegin
        rgx_replace = r'\\DIFdel\{((?:\\}|[^\}])*)\} *\\DIFadd\{((?:\\}|[^\}])*)\}'
        rgx_delete = r'\\DIFdel\{((?:\\}|[^\}])*)\}'
        rgx_add = r'\\DIFadd\{((?:\\}|[^\}])*)\}'

        # need to store and order the actions separately,
        # otherwise, parsing previous (long) actions may distort the relative order
        replace_actions = []
        delete_actions = []
        add_actions = []

        abstract = re.sub(rgx_replace, lambda m: _parse_abstract(m, 'R'), abstract)
        # print(abstract)
        # input('after R')
        abstract = re.sub(rgx_delete, lambda m: _parse_abstract(m, 'D'), abstract)
        # print('after D')
        # input(abstract)
        abstract = re.sub(rgx_add, lambda m: _parse_abstract(m, 'A'), abstract)
        # print('after A')
        # input(abstract)
        replace_actions = sorted(replace_actions)
        delete_actions = sorted(delete_actions)
        add_actions = sorted(add_actions)

        # normalize whitespaces
        abstract = ' '.join(abstract.split())
        # print(f'\n{abstract}')

        cursor = 0
        action_type = None
        before_revision = ''
        edit_actions = []
        while cursor < len(abstract):
            char = abstract[cursor]
            if char == self.anchor_symbol:
                assert abstract[cursor + 2] == self.anchor_symbol
                action_type = abstract[cursor + 1]
                assert action_type in set(['R', 'D', 'A'])

                if action_type in ['R', 'D']:
                    if action_type == 'R':
                        _, action = replace_actions.pop(0)
                    else:
                        _, action = delete_actions.pop(0)
                    before_part = action['before']
                    # assert before_part == abstract[cursor + 3:len(before_part)]
                    if before_part != abstract[cursor + 3:cursor + 3 + len(before_part)]:
                        print(f'ERROR: "{before_part}" != "{abstract[cursor + 3:cursor + 3 + len(before_part)]}"')
                    action['start_char_pos'] = len(before_revision)
                    before_revision = before_revision + before_part
                    action['end_char_pos'] = len(before_revision)
                    cursor = cursor + 3 + len(before_part)
                else:  # 'A'
                    _, action = add_actions.pop(0)
                    action['start_char_pos'] = len(before_revision)
                    action['end_char_pos'] = action['start_char_pos']
                    cursor = cursor + 3
                edit_actions.append(action)
            else:
                before_revision = before_revision + char
                cursor = cursor + 1

        if not self.sanity_check_before_revision(before_revision, edit_actions):
            print('ERROR: before_revision sanity check not passed!')
            return False, False, False

        # print(f'\n{before_revision}\n')

        after_revision = self.get_after_revision(before_revision, edit_actions)

        if not self.sanity_check_after_revision(after_revision):
            print('ERROR: after_revision sanity check not passed!')
            return False, False, False

        # print(f'\n{after_revision}\n')
        # print(edit_actions)

        return before_revision, after_revision, edit_actions

    def _sanity_check_remaining_latexdiff_command(self, text: str) -> bool:
        if 'DIFadd' in text or 'DIFdel' in text:
            print(f'ERROR: DIFadd or DIFdel in text:\n{text}')
            return False
        return True

    def _sanity_check_remaining_anchor(self, text: str) -> bool:
        if self.anchor_symbol in text:
            print(f'ERROR: anchor symbol {self.anchor_symbol} in text:\n{text}')
            return False
        return True

    def sanity_check_before_revision(self, text: str, edit_actions: List[Dict]) -> bool:
        if not self._sanity_check_remaining_latexdiff_command(text):
            return False
        if not self._sanity_check_remaining_anchor(text):
            return False
        is_passed = True
        for action in edit_actions:
            if action['before'] is not None:
                char_level_before = text[action['start_char_pos']:action['end_char_pos']]
                if char_level_before != action['before']:
                    print(
                        f'ERROR: char-level `before` not matched!\n"{char_level_before}" != "{action["before"]}"\n{action}')
                    is_passed = False
        return is_passed

    def sanity_check_after_revision(self, text: str) -> bool:
        if not self._sanity_check_remaining_latexdiff_command(text):
            return False
        if not self._sanity_check_remaining_anchor(text):
            return False
        return True

    def get_after_revision(self, before_revision: str, edit_actions: List[Dict]) -> str:
        """Construct after_revision from before_revision and edit_actions"""
        chunks = []
        edit_actions = sorted(edit_actions, key=lambda d: d['start_char_pos'])
        index = 0
        for action in edit_actions:
            chunk = before_revision[index:action['start_char_pos']]
            chunks.append(chunk)
            index = action['end_char_pos']
            if action['type'] == 'R':
                chunks.append(action['after'])
            elif action['type'] == 'D':
                pass
            else:
                chunks.append(action['after'])
        # add remaining chunk if any
        if index < len(before_revision) - 1:
            chunks.append(before_revision[index:])
        after_revision = ''.join(chunks)
        after_revision = re.sub(r' {2,}', ' ', ''.join(chunks))
        return after_revision

    def split_and_align(self, sample: Dict):
        def _handle_each_action(whole_sentence: str, sentences: List[str], action: Dict, char_to_sent: Dict, sent_to_char: Dict) -> Dict:
            start_char_pos = action['start_char_pos']
            end_char_pos = action['end_char_pos']
            start_sent_index = char_to_sent[start_char_pos]
            end_sent_index = char_to_sent[end_char_pos - 1] if end_char_pos > 0 else 0
            # if start_sent_index != end_sent_index:
            #     print(f'Sentences are merged, {start_sent_index}:{end_sent_index}')
            before_sentence = ''.join(sentences[start_sent_index:end_sent_index + 1])

            start_char_pos_sent_level = start_char_pos - sent_to_char[start_sent_index]
            end_char_pos_sent_level = end_char_pos - sent_to_char[start_sent_index]
            before_revision_part = before_sentence[start_char_pos_sent_level:end_char_pos_sent_level]
            assert before_revision_part == whole_sentence[start_char_pos:end_char_pos]
            action['start_char_pos_sent_level'] = start_char_pos_sent_level
            action['end_char_pos_sent_level'] = end_char_pos_sent_level
            action['start_sent_index'] = start_sent_index
            action['end_sent_index'] = end_sent_index
            after_sentence = get_after_revision(before_sentence, action)
            action['before_sent'] = before_sentence
            action['after_sent'] = after_sentence
            action['sent_start_char_pos'] = sent_to_char[start_sent_index]
            return action

        def get_after_revision(before_revision: str, edit_action: Dict) -> str:
            """Construct after_revision from before_revision and edit_action"""
            chunks = [before_revision[:edit_action['start_char_pos_sent_level']]]
            if edit_action['type'] == 'R':
                chunks.append(edit_action['after'])
            elif edit_action['type'] == 'D':
                pass
            else:
                chunks.append(edit_action['after'])
            chunks.append(before_revision[edit_action['end_char_pos_sent_level']:])
            after_revision = ''.join(chunks)
            after_revision = re.sub(r' {2,}', ' ', ''.join(chunks))
            return after_revision

        def _split_and_align(sample: Dict) -> List[Dict]:
            before = sample['before_revision']
            ic(before)
            before_sents = self.splitter.segment(before)
            ic(before_sents)
            char_to_sent = {}
            sent_to_char = {}
            cuml_char_index = 0
            for sent_index, sent in enumerate(before_sents):
                sent_to_char[sent_index] = cuml_char_index
                for _ in range(len(sent)):
                    char_to_sent[cuml_char_index] = sent_index
                    cuml_char_index += 1
            char_to_sent[cuml_char_index] = sent_index

            edit_actions = sample['edit_actions']
            sent_level_actions = []
            for action in edit_actions:
                sent_level_action = _handle_each_action(before, before_sents, action, char_to_sent, sent_to_char)
                sent_level_actions.append(sent_level_action)
            return sent_level_actions

        sent_level_actions = _split_and_align(sample)
        return sent_level_actions



def fuzzy_search(text, query, rough_start, rough_end, avoid_mid_word_cuts=False):
    query_length = len(query)
    matches = []

    #for i in range(len(text) - query_length + 1):
    #    # Check for word boundaries if needed
    #    if avoid_mid_word_cuts:
    #        if i > 0 and text[i-1].isalnum():
    #            # Current start is in the middle of a word
    #            continue
    #        if i + query_length < len(text) and text[i + query_length].isalnum():
    #            # Current end is in the middle of a word
    #            continue

    #    # Extract and compare a potential match
    #    potential_match = text[i:i + int(query_length *1.05)]
    #    similarity = difflib.SequenceMatcher(None, potential_match, query).ratio()
    #    matches.append((similarity, i, i + query_length, potential_match))
    #
    #print(f'matches: {matches}')
    #if matches:
    #    closest_match = max(
    #        matches,
    #        key=lambda x: (x[0], -abs(x[1] - rough_start) - abs(x[2] - rough_end))
    #    )
    #    return closest_match[1], closest_match[2], closest_match[3]
    #if query in text and query not in text[text.index(query)+len(query):]:
    #    rough_start = text.index(query)
    #    rough_end = rough_start + len(query)

    #if query != text[rough_start:rough_end]:
    #    try:
    #        query_start_token = query.split()[0]
    #        query_end_token = query.split()[-1]

    #        is_unique_query_start_token = False
    #        is_unique_query_end_token = False
    #        if query_start_token in text and query_start_token not in text[text.index(query_start_token)+len(query_start_token):]:
    #            rough_start = text.index(query_start_token)
    #            is_unique_query_start_token = True
    #        if query_end_token in text and query_end_token not in text[text.index(query_end_token)+len(query_end_token):]:
    #            rough_end = text.index(query_end_token) + len(query_end_token)
    #            is_unique_query_end_token = True

    #        if is_unique_query_start_token and is_unique_query_end_token:
    #            print("Closest match start:", rough_start)
    #            print("Closest match end:", rough_end)
    #            print("Closest match:", text[rough_start:rough_end])
    #            return rough_start, rough_end, text[rough_start:rough_end]

    #        else:
    #            matches = regex.finditer(r'(?b)({})'.format(query_start_token) + '{e}', text)

    #            filtered_matches = []
    #            for match in matches:
    #                filtered_matches.append((match.start(), match.end(), match.group(0)))

    #            if filtered_matches:
    #                closest_match = min(
    #                    filtered_matches,
    #                    key=lambda x: (abs(x[0] - rough_start))
    #                )
    #            closest_match_start = closest_match[0]


    #            matches = regex.finditer(r'(?b)({})'.format(query_end_token) + '{e}', text)

    #            filtered_matches = []
    #            for match in matches:
    #                filtered_matches.append((match.start(), match.end(), match.group(0)))

    #            if filtered_matches:
    #                closest_match = min(
    #                    filtered_matches,
    #                    key=lambda x: (abs(x[0] - rough_end))
    #                )
    #            closest_match_end = closest_match[0]

    #            print("Closest match start:", closest_match_start)
    #            print("Closest match end:", closest_match_end+len(query_end_token))
    #            print("Closest match:", text[closest_match_start:closest_match_end+len(query_end_token)])
    #            return closest_match_start, closest_match_end+len(query_end_token), text[closest_match_start:closest_match_end+len(query_end_token)]
    #    except Exception as e:
    #        print(f'Error: {e}')
    #        print('Fuzzy search failed')
    #        print('Query:', query)
    #        print('Text:', text)

    #        return None
    #else:
    #    return rough_start, rough_end, text[rough_start:rough_end]


    try:
        if query == text[rough_start:rough_end]:
            print("Everything is fine; Nothing to do.")
            print("Start:", rough_start)
            print("End:", rough_end)
            print("Match:", query)
            return rough_start, rough_end, query
        else:
            if query in text:
                if query not in text[text.index(query)+len(query):]:
                    start = text.index(query)
                    end = start + len(query)
                    print("Query found exactly once in text.")
                    print("Start:", start)
                    print("End:", end)
                    print("Match:", query)
                    return start, end, query
                else:
                    print("Query found more than once in text.")
                    matches = re.finditer(re.escape(query), text)
                    closest_match = min(
                        matches,
                        key=lambda x: (abs(x.start() - rough_start), abs(x.end() - rough_end))
                    )
                    start = closest_match.start()
                    end = closest_match.end()
                    print("Start:", start)
                    print("End:", end)
                    print("Match:", text[start:end])
                    return start, end, text[start:end]
            else:
                print("Query not found in text.")
                tmp_start_substring = query[0]
                for i in range(1, len(query)):
                    if tmp_start_substring in text:
                        if tmp_start_substring not in text[text.index(tmp_start_substring)+len(tmp_start_substring):]:
                            start = text.index(tmp_start_substring)
                            break
                        else:
                            tmp_start_substring = query[:i+1]
                    else:
                        closest_start = min(
                                re.finditer(re.escape(tmp_start_substring[:-1]), text),
                            key=lambda x: (abs(x.start() - rough_start))
                            )
                        start = closest_start.start()

                tmp_end_substring = query[-1]
                for i in range(1, len(query)):
                    if query[-i:] in text:
                        if query[-i:] not in text[text.index(query[-i:])+len(query[-i:]):]:
                            end = text.index(query[-i:]) + len(query[-i:])
                            break
                        else:
                            tmp_end_substring = query[-i:]
                    else:
                        closest_end = min(
                            re.finditer(re.escape(tmp_end_substring[1:]), text),
                            key=lambda x: (abs(x.end() - rough_end))
                        )
                        end = closest_end.end()
                print("Start:", start)
                print("End:", end)
                print("Match:", text[start:end])
                print(tmp_start_substring[:-1])
                print(tmp_end_substring[1:])
                return start, end, text[start:end]
    except Exception as e:
        print("Error:", e)
        print("Start:", rough_start)
        print("End:", rough_end)
        print("Match:", text[rough_start:rough_end])
        return None
    return None

def fuzzy_post_process_edits(edits):
    print(edits)
    final_edits = []
    for edit in edits:
        for i, edit_action in enumerate(edit['edit_actions']):
            if edit_action['type'] == 'R' or edit_action['type'] == 'D':
                if edit_action['type'] == 'R':
                    edit_action['after'] = LatexNodes2Text().latex_to_text(edit_action['after'])
                edit_action['before'] = LatexNodes2Text().latex_to_text(edit_action['before'])
                print(edit['before_revision'])
                print(edit_action['before'])
                print(edit_action['start_char_pos'])
                print(edit_action['end_char_pos'])
                fuzzy_result = fuzzy_search(edit['before_revision'], edit_action['before'], int(edit_action['start_char_pos']), int(edit_action['end_char_pos']), avoid_mid_word_cuts=False)
                if fuzzy_result is None:
                    print('Fuzzy search failed')
                    continue
                else:
                    fuzzy_start, fuzzy_end, fuzzy_text = fuzzy_result
                edit_action['start_char_pos'] = fuzzy_start
                edit_action['end_char_pos'] = fuzzy_end
                edit_action['before'] = fuzzy_text
                print(edit_action['before'])
                print(edit_action['start_char_pos'])
                print(edit_action['end_char_pos'])
                print('-'*20)
            elif edit_action['type'] == 'A':
                edit_action['after'] = LatexNodes2Text().latex_to_text(edit_action['after'])
                fuzzy_result= fuzzy_search(edit['before_revision'], ' ', int(edit_action['start_char_pos']), int(edit_action['end_char_pos']))
                if fuzzy_result is None:
                    print('Fuzzy search failed')
                    continue
                else:
                    fuzzy_start, fuzzy_end, fuzzy_text = fuzzy_result
                edit_action['start_char_pos'] = fuzzy_start
                edit_action['end_char_pos'] = fuzzy_end

            edit_obj = {"reason": "Other Reasons"}
            if edit_action['type'] == 'R':
                edit_obj["inappropriate_part"] = edit_action['before']
                edit_obj["rewritten_part"] = edit_action['after']
            elif edit_action['type'] == 'A':
                edit_obj["inappropriate_part"] = ""
                edit_obj["rewritten_part"] = edit_action['after']
            elif edit_action['type'] == 'D':
                edit_obj["inappropriate_part"] = edit_action['before']
                edit_obj["rewritten_part"] = ""

            if edit_obj not in final_edits:
                 final_edits.append(edit_obj)

    return {"edits": final_edits}, edit['before_revision']

if __name__ == '__main__':
    example = {"doc_id": "224", "version_depth": 1, "before_revision": "for everyone who is talking about RAPE in this subject let me ask you one thing!!!! if you got in a huge fight with someone and ended up breaking your hand or arm... would you cut it off just because it would REMIND you of that expirience???\r\nif your actualy SANE you would say no and if you say yes you need to see a Physiatrist!!!!", "after_revision": "For everyone who is discussing RAPE in this topic, let me ask you one thing. If you got into a huge fight with someone and ended up breaking your hand or arm, would you cut it off just because it would remind you of that experience? Of course not, if you're sane, you would know that wouldn't be a logical or healthy solution. If you do think about it, then you need to see a psychiatrist.\n"}
    parser = DirectLatexdiffParser()
    parsed_example = parser.parse_latex_diff(example['before_revision'], example['after_revision'], '../temp')
    ic(parsed_example)
    ic(fuzzy_post_process_edits([parsed_example]))

    df1 = pd.read_csv('/mnt/home/tziegenb/appropriateness-feedback/src/annotation-interface/appropriateness-study-abs/data/study_edits_part1.csv')
    df2 = pd.read_csv('/mnt/home/tziegenb/appropriateness-feedback/src/annotation-interface/appropriateness-study-abs/data/study_edits_part2.csv')
    df = pd.concat([df1, df2], ignore_index=True)
    ic(df.head())

    eval_dataset = load_dataset("timonziegenbein/appropriateness-corpus", split="validation")
    # extrac all from df that is in eval_dataset based on ids
    eval_ids = set(eval_dataset['post_id'])
    df = df[df['id'].isin(eval_ids)].sort_values(by=['id']).reset_index(drop=True)
    ic(df.head())
    ic(len(df))

    for model in ['rewrite_40a_60ss', 'rewrite_instruct', 'rewrite_50a_50ss', 'rewrite_60a_40ss', 'rewrite_10a_00ss']:
        edits = []
        for i, row in df.iterrows():
            print(f'Processing {i+1}/{len(df)}')
            parsed_example = parser.parse_latex_diff(row['source'], row[model], '../temp')
            edits.append(parsed_example)
            ic(row['issue'])
            break
        json_output, before_revision = fuzzy_post_process_edits(edits)
        ic(json_output)
