
import os

import torch
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker


class Ckip:  # step 2: 文章斷字斷詞
    def __init__(self):
        # 匯入停用詞
        dir_path = os.path.dirname(os.path.abspath(__file__))
        with open(f"{dir_path}/stopwords_TW.txt", "r", encoding="utf-8") as f:
            self.stopwords = [word.strip("\n") for word in f.readlines()]

        ## Initialize drivers (離線) 
        device = 0 if torch.cuda.is_available() else -1 # 應對 MacOS 不能使用 CUDA 的問題
        self.ws_driver = CkipWordSegmenter(model_name="./ckip_model/bert-base-chinese-ws", device=device)
        self.pos_driver = CkipPosTagger(model_name="./ckip_model/bert-base-chinese-pos", device=device)
        self.ner_driver = CkipNerChunker(model_name="./ckip_model/bert-base-chinese-ner", device=device)

    # 對文章進行斷詞
    def do_CKIP_WS(self, article):
        ws_results = self.ws_driver([str(article)])
        return ws_results
    
    # 對詞組進行詞性標示
    def do_CKIP_POS(self, ws_result):
        pos = self.pos_driver(ws_result[0])
        all_list = []
        for sent in pos:
            all_list.append(sent)
        return all_list
    
    # 保留名詞與動詞
    def _pos_filter(self, pos):
        for i in list(set(pos)):
            if i.startswith("N") or i.startswith("V"):
                return "Yes"
            else:
                continue
    
    # 去除數字與網址詞組
    def _remove_number_url(self, ws):
        number_pattern = "^\d+\.?\d*"
        url_pattern = "^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$"
        space_pattern = "^ .*"
        num_regex = re.match(number_pattern, ws)
        url_regex = re.match(url_pattern, ws)
        space_regex = re.match(space_pattern, ws)
        if num_regex ==  None and url_regex == None and space_regex == None:
            return True
        else:
            return False
    
    # 執行資料清洗
    def cleaner(self, ws_results, pos_results):
        stopwords = self.stopwords
        word_lst = []
        for ws, pos in zip(ws_results[0], pos_results):
            in_stopwords_or_not = ws not in stopwords          #詞組是否存為停用詞
            if_len_greater_than_1 = len(ws) >= 2                #詞組長度必須大於1
            is_V_or_N = self._pos_filter(pos)                        #詞組是否為名詞、動詞
            is_num_or_url = self._remove_number_url(ws)              #詞組是否為數字、網址、空白開頭
            if in_stopwords_or_not and if_len_greater_than_1 and is_V_or_N == "Yes" and is_num_or_url:
                word_lst.append(str(ws))
            else:
                pass
        return word_lst

ckip = Ckip()
