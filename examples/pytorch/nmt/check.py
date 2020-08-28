from stanfordcorenlp import StanfordCoreNLP


processor = StanfordCoreNLP('http://localhost', port=9000, timeout=1000)
props = {
            'annotators': 'ssplit,tokenize,depparse',
            "tokenize.options":
                "splitHyphenated=false,normalizeParentheses=true,normalizeOtherBrackets=true",
            "tokenize.whitespace": False,
            'ssplit.isOneSentence': False,
            'outputFormat': 'json'
        }
raw_text_data = "James went to the corner-shop."
dep_json = processor.annotate(raw_text_data.strip(), properties=props)
print(dep_json)