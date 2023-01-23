import torch
from torch import nn
import torch.nn.functional as F
from modules.transformer import TransformerEncoder




class ATmodel(nn.Module):
    def __init__(self, model_args):
        
        """
        Cross-modality
        """
        super(ATmodel, self).__init__()
        
        # Model Hyperparameters
        self.num_heads = model_args.num_heads
        self.layers = model_args.layers
        self.attn_mask = model_args.attn_mask
        output_dim = model_args.output_dim
        self.a_dim, self.v_dim = 256, 50
        self.attn_dropout = model_args.attn_dropout
        self.relu_dropout = model_args.relu_dropout
        self.res_dropout = model_args.res_dropout
        self.out_dropout = model_args.out_dropout
        self.embed_dropout = model_args.embed_dropout
        self.d_v = 50
        self.hidden_1 = 120
        self.hidden_2 = 120
        
        combined_dim = 2*self.d_v

        # 1D convolutional projection layers
        self.conv_1d_a = nn.Conv1d(self.a_dim, self.d_v, kernel_size=1, padding=0, bias=False)
        self.conv_1d_a = nn.Conv1d(self.a_dim, self.d_v, kernel_size=1, padding=0, bias=False)

        # Self Attentions 
        self.a_mem = self.transformer_arch(self_type='audio_self')
        self.v_mem = self.transformer_arch(self_type='text_self')
        
        self.trans_a_mem = self.transformer_arch(self_type='audio_self', scalar = True)
        self.trans_v_mem = self.transformer_arch(self_type='text_self', scalar = True)
        
        # Cross-modal 
        self.trans_v_with_a = self.transformer_arch(self_type='text/audio', pos_emb = True)
        self.trans_a_with_v = self.transformer_arch(self_type='audio/text', pos_emb = True)
       
        # Auxiliary networks linear layers
        self.proj_aux1 = nn.Linear(self.d_v, self.hidden_2)
        self.proj_aux2 = nn.Linear(self.hidden_2, self.hidden_1)
        self.proj_aux3 = nn.Linear(self.hidden_1, self.d_v)
        self.out_layer_aux = nn.Linear(self.d_v, output_dim)
        
        # Linear layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)


    def transformer_arch(self, self_type='audio/text', scalar = False, pos_emb = False):
        if self_type == 'text/audio':
            embed_dim, attn_dropout = self.d_v, 0
        elif self_type == 'audio/text':
            embed_dim, attn_dropout = self.d_v, 0
        elif self_type == 'audio_self':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout    
        elif self_type == 'text_self':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Not a valid network")
        
        return TransformerEncoder(embed_dim = embed_dim,
                                  num_heads = self.num_heads,
                                  layers = self.layers,
                                  attn_dropout = attn_dropout,
                                  relu_dropout = self.relu_dropout,
                                  res_dropout = self.res_dropout,
                                  embed_dropout = self.embed_dropout,
                                  attn_mask = self.attn_mask,
                                  scalar = scalar,
                                  pos_emb = pos_emb)
    
    
    def forward(self, x_aud, x_vid):
        """
        audio, and text should have dimension [batch_size, seq_len, n_features]
        """     
        

        x_aud = x_aud.transpose(1, 2)
        x_vid = x_vid.transpose(1, 2)


       
        # 1-D Convolution text/audio features
        proj_a_v = x_aud if self.a_dim == self.d_v else self.conv_1d_a(x_aud)
        proj_x_a = proj_a_v.permute(2, 0, 1)
        proj_x_v = x_vid.permute(2, 0, 1)
  
        # Audio/text
        h_av = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = self.trans_a_mem(h_av)
        representation_audio = h_as[-1]

    
        # text/Audio
        h_va = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = self.trans_v_mem(h_va)
        representation_text = h_vs[-1]
    
        # Concatenating audio-textual representations
        av_h_rep = torch.cat([representation_audio, representation_text], dim=1)
        
        
        # audio network
        h_a1 = self.a_mem(proj_x_a)
        h_a2 = self.a_mem(h_a1)
        h_a3 = self.a_mem(h_a2)
        h_rep_a_aux = h_a3[-1]   
            
        # text network
        h_v1 = self.v_mem(proj_x_v)
        h_v2 = self.v_mem(h_v1)
        h_v3 = self.v_mem(h_v2)
        h_rep_v_aux = h_v3[-1]
            
        #Audio  network output
        linear_hs_proj_a = self.proj_aux3(F.dropout(F.relu(self.proj_aux2(F.dropout(F.relu(self.proj_aux1(h_rep_a_aux)), p=self.out_dropout, training=self.training))), p=self.out_dropout, training=self.training))
        linear_hs_proj_a += h_rep_a_aux
        output_a_aux = self.out_layer_aux(linear_hs_proj_a)
        
        #text  network output
        linear_hs_proj_v = self.proj_aux3(F.dropout(F.relu(self.proj_aux2(F.dropout(F.relu(self.proj_aux1(h_rep_v_aux)), p=self.out_dropout, training=self.training))), p=self.out_dropout, training=self.training))
        linear_hs_proj_v += h_rep_v_aux
        output_v_aux = self.out_layer_aux(linear_hs_proj_v)
        
        #Main network output
        linear_hs_proj_av = self.proj2(F.dropout(F.relu(self.proj1(av_h_rep)), p=self.out_dropout, training=self.training))
        linear_hs_proj_av += av_h_rep
        output = self.out_layer(linear_hs_proj_av)
        
        
        return output, output_a_aux, output_v_aux