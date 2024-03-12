import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_len, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, input_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_dim, 2).float()* (-math.log(10000.0) / input_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = pe
        
    def forward(self, x):
        self.pe = self.pe.to(x.device)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    
class TemporalConvolutionalLayer(nn.Module):
    def __init__(self, input_dim, output_dim, max_len, dropout):
        super(TemporalConvolutionalLayer, self).__init__()
        self.conv1d = nn. Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, padding=1)
        self.pos_encoding = PositionalEncoding(input_dim, max_len, dropout)
        
    def forward(self, x, mask):
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoding(x)
        return x
        
# Currently not used     
class CrossModalAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(CrossModalAttention, self).__init__()
        self.num_heads = num_heads
        self.W_q = nn.Linear(input_dim // self.num_heads, input_dim // self.num_heads)
        self.W_k = nn.Linear(input_dim  // self.num_heads, input_dim // self.num_heads)
        self.W_v = nn.Linear(input_dim  // self.num_heads, input_dim // self.num_heads)
        self.W_ff = nn.Linear(input_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, Zi_1_alpha_beta, Zi_1_alpha):
        Z_i_alpha_beta = self.layer_norm(Zi_1_alpha_beta)
        Z_i_alpha = self.layer_norm(Zi_1_alpha)
        
        dim_per_head = Z_i_alpha.size(2) // self.num_heads
        batch_size = Z_i_alpha.size(0)
        seq_len = Z_i_alpha.size(1)
        embed_dim = Z_i_alpha.size(2)
        
        q = Z_i_alpha_beta.view(batch_size, seq_len, self.num_heads, dim_per_head)
        k = Z_i_alpha.view(batch_size, seq_len, self.num_heads, dim_per_head)
        v = Z_i_alpha.view(batch_size, seq_len, self.num_heads, dim_per_head)
        
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)
       
        k = k.permute(0, 2, 1, 3)
        attention_scores = torch.matmul(q, k) / dim_per_head
        attention_weights = torch.nn.functional.softmax(attention_scores / math.sqrt(Z_i_alpha.size(-1)), dim=-1)

        attended_representation = torch.matmul(attention_weights, v)
        attended_representation = attended_representation.view(batch_size, seq_len, embeded_dim)
        cross_modal_adaption = attended_representation + Z_i_alpha_beta
        
        z_i_alpha_to_beta_intermediate = self.layer_norm(cross_modal_adaption)
        
        z_i_alpha_to_beta_ff = self.W_ff(z_i_alpha_to_beta_intermediate)
        z_i_alpha_to_beta_final = self.layer_norm(z_i_alpha_to_beta_intermediate + z_i_alpha_to_beta_ff)
        
        return z_i_alpha_to_beta_final, attended_representation

class CrossModalAttention2(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(CrossModalAttention2, self).__init__()
        self.num_heads = num_heads
        self.multihead_attention = nn.MultiheadAttention(input_dim, num_heads)
        self.W_ff = nn.Linear(input_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, Zi_1_alpha_beta, Zi_1_alpha):
        Z_i_alpha_beta = self.layer_norm(Zi_1_alpha_beta)
        Z_i_alpha = self.layer_norm(Zi_1_alpha)

        cross_modal_adaption, attn_output_weights = self.multihead_attention(
            Z_i_alpha_beta.permute(1, 0, 2),
            Z_i_alpha.permute(1, 0, 2),
            Z_i_alpha.permute(1, 0, 2)
        )

        cross_modal_adaption = cross_modal_adaption.permute(1, 0, 2)
        cross_modal_adaption = cross_modal_adaption + Zi_1_alpha_beta

        z_i_alpha_to_beta_intermediate = self.layer_norm(cross_modal_adaption)
        z_i_alpha_to_beta_ff = self.W_ff(z_i_alpha_to_beta_intermediate)
        z_i_alpha_to_beta_final = self.layer_norm(z_i_alpha_to_beta_intermediate + z_i_alpha_to_beta_ff)

        return z_i_alpha_to_beta_final, attn_output_weights

        
class MultimodalCrossTransformer(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultimodalCrossTransformer, self).__init__()
        self.audio_to_video_attention = CrossModalAttention2(input_dim, num_heads)
        self.text_to_video_attention = CrossModalAttention2(input_dim, num_heads)
        
        self.video_to_audio_attention = CrossModalAttention2(input_dim, num_heads)
        self.text_to_audio_attention = CrossModalAttention2(input_dim, num_heads)
        
        self.video_to_text_attention = CrossModalAttention2(input_dim, num_heads)
        self.audio_to_text_attention = CrossModalAttention2(input_dim, num_heads)
        
    def forward(self, video, audio, text):
        # Video Modality
        audio_to_video, av_attention_scores = self.audio_to_video_attention(audio, video)
        text_to_video, tv_attention_scores = self.text_to_video_attention(text, video)
        video_attention  = torch.cat([audio_to_video, text_to_video], dim=-1)
        
        # Audio Modality
        video_to_audio, va_attention_scores = self.video_to_audio_attention(video, audio)
        text_to_audio, ta_attention_scores = self.text_to_audio_attention(text, audio)
        audio_attention = torch.cat([video_to_audio, text_to_audio], dim=-1)
        
        # Text Modality
        video_to_text, vt_attention_scores = self.video_to_text_attention(video, text)
        audio_to_text, at_attention_scores = self.audio_to_text_attention(audio, text)
        text_attention = torch.cat([video_to_text, audio_to_text], dim=-1)
        
        # Dict to store attention scores
        attention_scores = {"av_attention_scores": av_attention_scores,
                           "tv_attention_scores": tv_attention_scores,
                           "va_attention_scores": va_attention_scores,
                           "ta_attention_scores": ta_attention_scores,
                           "vt_attention_scores": vt_attention_scores,
                           "at_attention_scores": at_attention_scores}
        
        return [video_attention, audio_attention, text_attention], attention_scores
        
        
class TemporalEnsemble(nn.Module):
    def __init__(self, num_layers, input_dim):
        super(TemporalEnsemble, self).__init__()
        
        self.transformers = nn.ModuleList([nn.Transformer(d_model=input_dim*2, nhead=3, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dim_feedforward=input_dim*4) for _ in range(3)])
        self.feedforward = nn.Linear(input_dim*6, input_dim) # change to simple MLP with Relu?
        
    def forward(self, hidden_states_list):
        num_modalities = len(hidden_states_list)
        temporal_representations = []
        
        for modality_idx, hidden_states in enumerate(hidden_states_list):
            modality_transformer = self.transformers[modality_idx]
            temporal_rep = modality_transformer(hidden_states, hidden_states)
            temporal_representations.append(temporal_rep)
        
        concatenated_temporal_rep = torch.cat(temporal_representations, dim=-1)
        output = self.feedforward(concatenated_temporal_rep)
        return output
    
    
class ModalitySpecificAttentionFusion(nn.Module):
    def __init__(self, input_dim):
        super(ModalitySpecificAttentionFusion, self).__init__()
        
        self.W_prime = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, representation):
        W_prime_alpha = self.W_prime(representation)
        normalized_attention_weight = self.softmax(W_prime_alpha)
        
        fused_representation = normalized_attention_weight * representation
        
        return fused_representation
    
    
class TemporalEnsembleWithFusion(nn.Module):
    def __init__(self, input_dim):
        super(TemporalEnsembleWithFusion, self).__init__()
        
        # Feed forward layer for fusion
        self.feed_forward_fusion = nn.Sequential(
            nn.Linear(input_dim, input_dim), 
            nn.ReLU(),
            nn.Linear(input_dim, input_dim) 
        )
        
    def forward(self, Z, Z_fused):
        Z_fused_ff = self.feed_forward_fusion(Z_fused)
        combined_representation = Z + Z_fused_ff
        
        return combined_representation

class MultimodalModel(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, num_layers, num_heads, max_len, dropout, movement):
        super(MultimodalModel, self).__init__()
        
        self.movement = movement
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        
        self.audio_conv1d_pe = TemporalConvolutionalLayer(embedding_dim[0], embedding_dim[0], max_len, dropout)

        self.modality_specific_self_attention = ModalitySpecificAttentionFusion(embedding_dim[0])
        
        # maybe add more layers according to num layers
        if not self.movement:
            # Asset price volatility prediction
            self.asset_price_volatility_mlp_index_large = nn.Sequential(
                nn.Linear(embedding_dim[0]*max_len, hidden_dim),
                nn.Linear(hidden_dim, 1)
            )
            self.asset_price_volatility_mlp_index_small = nn.Sequential(
                nn.Linear(embedding_dim[0]*max_len, hidden_dim),
                nn.Linear(hidden_dim, 1)
            )
            self.asset_price_volatility_mlp_gold = nn.Sequential(
                nn.Linear(embedding_dim[0]*max_len, hidden_dim),
                nn.Linear(hidden_dim, 1)
            )
            self.asset_price_volatility_mlp_dollar = nn.Sequential(
                nn.Linear(embedding_dim[0]*max_len, hidden_dim),
                nn.Linear(hidden_dim, 1)
            )
            self.asset_price_volatility_mlp_10_y_bond = nn.Sequential(
                nn.Linear(embedding_dim[0]*max_len, hidden_dim),
                nn.Linear(hidden_dim, 1)
            )
            self.asset_price_volatility_mlp_3_m_bond = nn.Sequential(
                nn.Linear(embedding_dim[0]*max_len, hidden_dim),
                nn.Linear(hidden_dim, 1)
            )
        
        else:
            # Asset price movement prediction
            self.asset_price_movement_mlp_index_large = nn.Sequential(
                nn.Linear(embedding_dim[0]*max_len, hidden_dim),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid() # Change Sigmoid to Relu, ...
            )
            self.asset_price_movement_mlp_index_small = nn.Sequential(
                nn.Linear(embedding_dim[0]*max_len, hidden_dim), 
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            self.asset_price_movement_mlp_gold = nn.Sequential(
                nn.Linear(embedding_dim[0]*max_len, hidden_dim),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            self.asset_price_movement_mlp_dollar = nn.Sequential(
                nn.Linear(embedding_dim[0]*max_len, hidden_dim),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            self.asset_price_movement_mlp_10_y_bond = nn.Sequential(
                nn.Linear(embedding_dim[0]*max_len, hidden_dim), 
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            self.asset_price_movement_mlp_3_m_bond = nn.Sequential(
                nn.Linear(embedding_dim[0]*max_len, hidden_dim),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

    def forward(self, video, audio, text, mask, subclip_mask):
        audio_conv1d_pe = self.audio_conv1d_pe(audio, mask)
        
        self_attention, att_weights = self.modality_specific_self_attention(audio_conv1d_pe)
        
        # Just helper so code below does not need to be changed. Actually no combined representation
        combined_representation = self_attention
    
        padded_flat_input = torch.zeros(combined_representation.size(0), self.max_len * combined_representation.size(2), device=combined_representation.device)
        padded_flat_input[:, :combined_representation.size(1) * combined_representation.size(2)] = (combined_representation * mask.unsqueeze(-1)).view(combined_representation.size(0), -1)
        
        # MLPs for Asset Prices
        if not self.movement:
            index_large_output = self.asset_price_volatility_mlp_index_large(padded_flat_input)
            index_small_output = self.asset_price_volatility_mlp_index_small(padded_flat_input)
            gold_output = self.asset_price_volatility_mlp_gold(padded_flat_input)
            dollar_output = self.asset_price_volatility_mlp_dollar(padded_flat_input)
            ten_y_bond_output = self.asset_price_volatility_mlp_10_y_bond(padded_flat_input)
            three_m_bond_output = self.asset_price_volatility_mlp_3_m_bond(padded_flat_input)
        else:
            index_large_output = self.asset_price_movement_mlp_index_large(padded_flat_input)
            index_small_output = self.asset_price_movement_mlp_index_small(padded_flat_input)
            gold_output = self.asset_price_movement_mlp_gold(padded_flat_input)
            dollar_output = self.asset_price_movement_mlp_dollar(padded_flat_input)
            ten_y_bond_output = self.asset_price_movement_mlp_10_y_bond(padded_flat_input)
            three_m_bond_output = self.asset_price_movement_mlp_3_m_bond(padded_flat_input)
        
        # Helper so Code does not have to be changed
        attention_scores_cross_transformer = {"No attention scores provided for this model": torch.zeros(2, 3, 4)}
        
        return torch.cat([index_large_output, index_small_output, gold_output, dollar_output, ten_y_bond_output, three_m_bond_output],dim=1), attention_scores_cross_transformer, attention_scores_cross_transformer