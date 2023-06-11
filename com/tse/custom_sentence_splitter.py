import email
import logging

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import download

download('stopwords')
download('punkt')
download('brown')
download('averaged_perceptron_tagger')

from string import punctuation
import re
from pathlib import Path
from cleantext import clean  # pip install clean-text
from nltk.corpus import stopwords
from textblob import TextBlob

END_OF_SENTENCE = 'END_OF_SENTENCE'
MAX_DELETE_N_BEGIN_LINES = 2


class CustomSentenceSplitter(object):
    def __init__(self, ):
        self.custom_stop_words = set(stopwords.words('english') + list(punctuation))
        self.regexp1 = re.compile(
            r'copyright 20|free subscription|all content is ©|©|thanks so much|best wishes,|thanks & regards'
            r'|kind regards|awaiting your|awaiting for your|await your response')
        self.regexp2 = re.compile(r'cheers,|thank you|thanks,|regards|sincerely|thx|thnx |thanking you,')
        self.regexp3 = re.compile('^cheers|^thank you|^thanks|^regards|^sincerely|^thx|^thnx|^thanking you')
        self.clean_line_prefix = re.compile(r"^\W+")
        pass

    def __custom_clean(self, text):
        # Remove parenthesis text
        text = re.sub(r'\([^)]*\)|\[[^]]*\]|\{[^}]*\}|\<[^>]*\>', ' ', text)
        text = re.sub(r'( )+', ' ', text)
        text = text.replace('\xa0', '')
        text = text.replace('\\n', '\n')
        text = text.replace(' Rs.', ' Rs')
        text = text.replace(' rs.', ' rs')
        text = text.replace(' RS.', ' RS')
        # regex = re.compile('[^a-zA-Z,.@:]')
        # text = regex.sub(' ', text)

        # Handle linebreaks
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\r( )+\n', '\r\n', text)
        text.replace("\r\n", END_OF_SENTENCE)
        text = re.sub(r'\n( )+\n', '\n\n', text)
        text = re.sub(r'\n( )+\n', '\n\n', text)
        text = re.sub(r'\n\n+', END_OF_SENTENCE, text)

        cleaned_text = ''
        text_parts = text.split(END_OF_SENTENCE)
        for text_part in text_parts:
            parts = text_part.split("\n")
            for part in parts:
                part = self.clean_line_prefix.sub("", part)
                p = part
                p = re.sub(r'[^A-Za-z]+', ' ', p)
                n_tokens = len(word_tokenize(p))
                if re.match(self.regexp3, part.lower()):
                    part = '\n' + part
                if n_tokens == 0:
                    part = part + '\n'
                elif n_tokens < 5:
                    part = part + END_OF_SENTENCE
                else:
                    part = part + '\n'
                cleaned_text = cleaned_text + part
            cleaned_text = cleaned_text + END_OF_SENTENCE
        cleaned_text = re.sub(r'\n\n+', END_OF_SENTENCE, cleaned_text)
        return cleaned_text

    def __clean_text_prefix(self, text):
        text = text.replace("\u2028", "\n")
        splits = text.split('\n')
        deleted_lines = 0
        is_delete = False
        for split in splits:
            p = self.clean_line_prefix.sub("", split)
            if p.lower().startswith('from:'):
                is_delete = True
                break
            deleted_lines += 1
        if is_delete and deleted_lines <= MAX_DELETE_N_BEGIN_LINES:
            text = text.split("\n", deleted_lines)[deleted_lines]
        return text

    def __get_mail_content(self, text):
        text = self.__clean_text_prefix(text)
        msg = email.message_from_string(text)
        content = msg.get_payload()

        return content, msg

    def __is_tail_sen(self, sen):
        s = sen.lower().strip()
        s = re.sub(r'[^A-Za-z]+', ' ', s)
        if self.regexp1.search(s):
            return True
        elif self.regexp2.search(s) and len(word_tokenize(s)) <= 4:
            return True
        elif self.regexp3.search(s) and len(word_tokenize(s)) <= 4:
            return True
        return False

    def __split_sentences(self, text):
        cleaned_text = self.__custom_clean(text)
        text_sentences = cleaned_text.split(END_OF_SENTENCE)
        sentences = []
        for s in text_sentences:
            s = s.replace('\n', ' ')
            if not self.__is_tail_sen(s):
                if len(re.sub('( )+', '', s)) > 20:
                    sens = sent_tokenize(s)
                    sentences.extend(sens)
                else:
                    sentences.append(s)
            else:
                break
        return sentences

    def __get_cleaned_sentences(self, text):
        sentences = self.__split_sentences(text)
        original_sentences = []
        sentence_tokens = []
        for sen in sentences:
            cleaned_sen = self.__clean_text_hard(sen).lower()
            tokens = [w for w in word_tokenize(cleaned_sen) if w not in self.custom_stop_words]
            if len(word_tokenize(sen)) > 3 and len(tokens) > 0:
                sentence_tokens.append(tokens)
                original_sentences.append(TextBlob(self.__clean_text_soft(sen)))
        return original_sentences, sentence_tokens

    def __clean_text_hard(self, text):
        x = clean(text,
                  fix_unicode=True,  # fix various unicode errors
                  to_ascii=True,  # transliterate to closest ASCII representation
                  lower=True,  # lowercase text
                  no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
                  no_urls=True,  # replace all URLs with a special token
                  no_emails=True,  # replace all email addresses with a special token
                  no_phone_numbers=True,  # replace all phone numbers with a special token
                  no_numbers=True,  # replace all numbers with a special token
                  no_digits=True,  # replace all digits with a special token
                  no_currency_symbols=True,  # replace all currency symbols with a special token
                  no_punct=True,  # fully remove punctuation
                  replace_with_url="",
                  replace_with_email="",
                  replace_with_phone_number="",
                  replace_with_number="",
                  replace_with_digit="0",
                  replace_with_currency_symbol="",
                  lang="en"
                  )
        return x

    def __clean_text_soft(self, text):
        return clean(text,
                     fix_unicode=True,  # fix various unicode errors
                     to_ascii=True,  # transliterate to closest ASCII representation
                     lower=False,  # lowercase text
                     no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
                     no_urls=False,  # replace all URLs with a special token
                     no_emails=False,  # replace all email addresses with a special token
                     no_phone_numbers=False,  # replace all phone numbers with a special token
                     no_numbers=False,  # replace all numbers with a special token
                     no_digits=False,  # replace all digits with a special token
                     no_currency_symbols=False,  # replace all currency symbols with a special token
                     no_punct=False,  # fully remove punctuation
                     replace_with_url="",
                     replace_with_email="",
                     replace_with_phone_number="",
                     replace_with_number="",
                     replace_with_digit="",
                     replace_with_currency_symbol="",
                     lang="en"  # set to 'de' for German special handling
                     )

    def get_sentences_from_mail(self, text):
        content, msg = self.__get_mail_content(text)
        original_sentences, sentence_tokens = self.__get_cleaned_sentences(content)
        return original_sentences, sentence_tokens, msg

    def __print_list(self, sentences):
        for s in sentences:
            logging.debug([s])
        pass


def main(**kwargs):
    path = '/home/muzaffar/mydata/quickwordz/data/mails/'
    path = path + 'Your payment to Ra-Wifi.txt'
    content = Path(path).read_text()
    splitter = CustomSentenceSplitter()
    original_sentences, sentence_tokens, msg = splitter.get_sentences_from_mail(content)
    email_from = msg['from']
    email_to = msg['to']
    email_date = msg['date']
    email_subject = msg['subject']
    print(email_to, email_from, email_date, email_subject)
    for sen in original_sentences:
        print([sen])


if __name__ == '__main__':
    main()
