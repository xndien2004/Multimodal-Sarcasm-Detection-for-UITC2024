import unicodedata
import re

class VietnameseTextPreprocessor:
    vowel_map = [
        ['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
        ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
        ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
        ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
        ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
        ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
        ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
        ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
        ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
        ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
        ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
        ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']
    ]

    tone_map = ['', 'f', 's', 'r', 'x', 'j']
    vowel_to_ids = {}

    @classmethod
    def initialize_vowel_to_ids(cls):
        for i in range(len(cls.vowel_map)):
            for j in range(len(cls.vowel_map[i]) - 1):
                cls.vowel_to_ids[cls.vowel_map[i][j]] = (i, j)

    @staticmethod
    def unicode_normalize(text):
        return unicodedata.normalize('NFC', text)

    @classmethod
    def is_valid_vietnamese_word(cls, word):
        chars = list(word)
        vowel_index = -1
        for index, char in enumerate(chars):
            x, y = cls.vowel_to_ids.get(char, (-1, -1))
            if x != -1:
                if vowel_index == -1:
                    vowel_index = index
                else:
                    if index - vowel_index != 1:
                        return False
                    vowel_index = index
        return True

    @classmethod
    def standardize_vietnamese_tone(cls, word):
        if not cls.is_valid_vietnamese_word(word):
            return word

        chars = list(word)
        tone = 0
        vowel_indices = []
        is_qu_or_gi = False
        for index, char in enumerate(chars):
            x, y = cls.vowel_to_ids.get(char, (-1, -1))
            if x == -1:
                continue
            elif x == 9 and index != 0 and chars[index - 1] == 'q':  # check 'qu'
                chars[index] = 'u'
                is_qu_or_gi = True
            elif x == 5 and index != 0 and chars[index - 1] == 'g':  # check 'gi'
                chars[index] = 'i'
                is_qu_or_gi = True
            if y != 0:
                tone = y
                chars[index] = cls.vowel_map[x][0]
            if not is_qu_or_gi or index != 1:
                vowel_indices.append(index)

        if len(vowel_indices) < 2:
            if is_qu_or_gi:
                if len(chars) == 2:
                    x, y = cls.vowel_to_ids.get(chars[1])
                    chars[1] = cls.vowel_map[x][tone]
                else:
                    x, y = cls.vowel_to_ids.get(chars[2], (-1, -1))
                    if x != -1:
                        chars[2] = cls.vowel_map[x][tone]
                    else:
                        chars[1] = cls.vowel_map[5][tone] if chars[1] == 'i' else cls.vowel_map[9][tone]
                return ''.join(chars)
            return word

        for index in vowel_indices:
            x, y = cls.vowel_to_ids[chars[index]]
            if x == 4 or x == 8:  # ê, ơ
                chars[index] = cls.vowel_map[x][tone]
                return ''.join(chars)

        if len(vowel_indices) == 2:
            if vowel_indices[-1] == len(chars) - 1:
                x, y = cls.vowel_to_ids[chars[vowel_indices[0]]]
                chars[vowel_indices[0]] = cls.vowel_map[x][tone]
            else:
                x, y = cls.vowel_to_ids[chars[vowel_indices[1]]]
                chars[vowel_indices[1]] = cls.vowel_map[x][tone]
        else:
            x, y = cls.vowel_to_ids[chars[vowel_indices[1]]]
            chars[vowel_indices[1]] = cls.vowel_map[x][tone]
        return ''.join(chars)


    @classmethod
    def standardize_sentence_tone(cls, sentence):
        sentence = sentence.lower()
        words = sentence.split()
        for index, word in enumerate(words):
            if not word:
                return " "
            cleaned_word = re.sub(r'(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)', r'\1/\2/\3', word).split('/')
            if len(cleaned_word) == 3:
                cleaned_word[1] = cls.standardize_vietnamese_tone(cleaned_word[1])
            words[index] = ''.join(cleaned_word)
        return ' '.join(words)

    @classmethod
    def fix_repeated_chars(cls, sentence): 
        return re.sub(r'(.)\1{2,}', r'\1', sentence)


    @classmethod
    def preprocess(cls, text):
        text = cls.unicode_normalize(text)
        text = cls.standardize_sentence_tone(text)
        text = cls.fix_repeated_chars(text)
        return text

