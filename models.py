from transformers import BertModel, BertConfig
from transformers import AutoModel
from torch import nn
import torch
from transformers.utils import logging
logging.set_verbosity_error()


class DE2EN(nn.Module):
    def __init__(self):
        super().__init__()

        conf = BertConfig.from_pretrained('bert-base-uncased')
        conf.is_decoder = True
        conf.add_cross_attention = True
        self.en = BertModel.from_pretrained('bert-base-uncased', config=conf, add_pooling_layer=False)
        self.en.encoder.layer = self.en.encoder.layer[:4]   # to reduce model size
        print('en #parameters:', sum(p.numel() for p in self.en.parameters() if p.requires_grad))

        self.de = AutoModel.from_pretrained("dbmdz/bert-base-german-uncased", add_pooling_layer=False)
        self.de.encoder.layer = self.de.encoder.layer[:4]   # to reduce model size
        print('de #parameters:', sum(p.numel() for p in self.de.parameters() if p.requires_grad))

        self.prediction_head = nn.Sequential(nn.Linear(768, 2048),
                                             nn.GELU(),
                                             nn.LayerNorm(2048, eps=1e-12),
                                             nn.Linear(2048, 30522)
                                             )

    def forward(self, de_input, en_input):
        label = en_input.input_ids.clone()
        de_output = self.de(de_input.input_ids, attention_mask=de_input.attention_mask,
                            return_dict=True).last_hidden_state   # batch*len_de*768
        en_output = self.en(en_input.input_ids, attention_mask=en_input.attention_mask,
                            encoder_hidden_states=de_output,
                            encoder_attention_mask=de_input.attention_mask,
                            return_dict=True).last_hidden_state   # batch*len_en*768
        output = self.prediction_head(en_output)    # batch*len_en*30522
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output.permute((0, 2, 1))[:, :, :-1], label[:, 1:])
        return loss

    def generate(self, de_embeds, de_mask, text_input):
        text_atts = torch.where(text_input == 0, 0, 1)
        token_output = self.en(text_input,
                               attention_mask=text_atts,
                               encoder_hidden_states=de_embeds,
                               encoder_attention_mask=de_mask,
                               return_dict=True).last_hidden_state
        token_output = self.prediction_head(token_output)[:, -1, :]
        token_output = torch.argmax(token_output, dim=-1)
        return token_output.unsqueeze(1)    # batch*1
