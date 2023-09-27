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
        #print("PositionalEncoding:")
        #print(x.size())
        self.pe = self.pe.to(x.device)
        x = x + self.pe[:x.size(0), :]
        #print(x.size())
        return self.dropout(x)
    
    
class TemporalConvolutionalLayer(nn.Module):
    def __init__(self, input_dim, output_dim, max_len, dropout):
        super(TemporalConvolutionalLayer, self).__init__()
        self.conv1d = nn. Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, padding=1)
        self.pos_encoding = PositionalEncoding(input_dim, max_len, dropout)
        
    def forward(self, x, mask):
        #x = x * mask.unsqueeze(-1)
        #print("TemporalConvolutional:")
        #print(x.size())
        x = x.squeeze(2)
        #print(x.size())
        x = x.permute(0, 2, 1)
        #print(x.size())
        x = self.conv1d(x)
        #print(x.size())
        x = x.permute(0, 2, 1)
        #print(x.size())
        x = self.pos_encoding(x)
        #print(x.size())
        return x
        
        
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
        #print("CrossmodalAttention:")
        #print("Zi_1_alpha_beta ", Zi_1_alpha_beta.size())
        #print("Zi_1_alpha ", Zi_1_alpha.size)
        Z_i_alpha_beta = self.layer_norm(Zi_1_alpha_beta)
        Z_i_alpha = self.layer_norm(Zi_1_alpha)
        
        dim_per_head = Z_i_alpha.size(2) // self.num_heads
        batch_size = Z_i_alpha.size(0)
        seq_len = Z_i_alpha.size(1)
        embed_dim = Z_i_alpha.size(2)
        
        q = Z_i_alpha_beta.view(batch_size, seq_len, self.num_heads, dim_per_head)
        k = Z_i_alpha.view(batch_size, seq_len, self.num_heads, dim_per_head)
        v = Z_i_alpha.view(batch_size, seq_len, self.num_heads, dim_per_head) # check if v might has to be Z_i_alpha_beta
        
        q = self.W_q(q)
        #print("q ", q.size())
        k = self.W_k(k)
        #print("k ", k.size())
        v = self.W_v(v)
        #print("v ", v.size())
       
        k = k.permute(0, 2, 1, 3)
        print(q.size())
        print(k.size())
        attention_scores = torch.matmul(q, k) / dim_per_head
        #print("attention_scores ", attention_scores.size())
        attention_weights = torch.nn.functional.softmax(attention_scores / math.sqrt(Z_i_alpha.size(-1)), dim=-1)
        #print("attention_weights ", attention_weights.size())

        attended_representation = torch.matmul(attention_weights, v)
        attended_representation = attended_representation.view(batch_size, seq_len, embeded_dim)
        #print("attended_representation: ", attended_representation.size())
        cross_modal_adaption = attended_representation + Z_i_alpha_beta
        #print("cross_modal_adaption ", cross_modal_adaption.size())
        
        z_i_alpha_to_beta_intermediate = self.layer_norm(cross_modal_adaption)
        #print("z_i_alpha_to_beta_intermediate ", z_i_alpha_to_beta_intermediate.size())
        
        z_i_alpha_to_beta_ff = self.W_ff(z_i_alpha_to_beta_intermediate)
        #print("z_i_alpha_to_beta_ff ", z_i_alpha_to_beta_ff.size())
        z_i_alpha_to_beta_final = self.layer_norm(z_i_alpha_to_beta_intermediate + z_i_alpha_to_beta_ff)
        #print("z_i_alpha_to_beta_final ", z_i_alpha_to_beta_final.size())
        
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
        #self.text_to_video_attention = CrossModalAttention2(input_dim, num_heads)
        
        self.video_to_audio_attention = CrossModalAttention2(input_dim, num_heads)
        #self.text_to_audio_attention = CrossModalAttention2(input_dim, num_heads)
        
        #self.video_to_text_attention = CrossModalAttention2(input_dim, num_heads)
        #self.audio_to_text_attention = CrossModalAttention2(input_dim, num_heads)
        
    def forward(self, video, audio):
        # Video Modality
        #print("MultimodalCrossTransformer: ")
        audio_to_video, av_attention_scores = self.audio_to_video_attention(audio, video)
        #print("audio_to_video", audio_to_video.size())
        #text_to_video, tv_attention_scores = self.text_to_video_attention(text, video)
        #print("text_to_video", text_to_video.size())
        #video_attention  = torch.cat([audio_to_video, text_to_video], dim=-1)
        #print("video_attention", video_attention.size())
        
        # Audio Modality
        video_to_audio, va_attention_scores = self.video_to_audio_attention(video, audio)
        #print("video_to_audio", video_to_audio.size())
        #text_to_audio, ta_attention_scores = self.text_to_audio_attention(text, audio)
        #print("text_to_audio", text_to_audio.size())
        #audio_attention = torch.cat([video_to_audio, text_to_audio], dim=-1)
        #print("audio_attention", audio_attention.size())
        
        # Text Modality
        #video_to_text, vt_attention_scores = self.video_to_text_attention(video, text)
        #print("video_to_text", video_to_text.size())
        #audio_to_text, at_attention_scores = self.audio_to_text_attention(audio, text)
        #print("audio_to_text", audio_to_text.size())
        #text_attention = torch.cat([video_to_text, audio_to_text], dim=-1)
        #print("text_attention", text_attention.size())
        
        # Dict to store attention scores
        attention_scores = {"ta_attention_scores": av_attention_scores,
                           "at_attention_scores": va_attention_scores}
        
        #return [audio_attention, text_attention], attention_scores
        return [audio_to_video, video_to_audio], attention_scores
        
        
class TemporalEnsemble(nn.Module):
    def __init__(self, num_layers, input_dim):
        super(TemporalEnsemble, self).__init__()
        self.input_dim = input_dim
        self.transformers = nn.ModuleList([nn.Transformer(d_model=input_dim, nhead=3, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dim_feedforward=input_dim*4) for _ in range(2)])
        self.feedforward = nn.Linear(input_dim*2, input_dim) # change to simple MLP with Relu?
        
    def forward(self, hidden_states_list):
        #print("TemporalEnsemble:")
        num_modalities = len(hidden_states_list)
        temporal_representations = []
        
        for modality_idx, hidden_states in enumerate(hidden_states_list):
            modality_transformer = self.transformers[modality_idx]
            temporal_rep = modality_transformer(hidden_states, hidden_states)
            #print("temporal_rep ", temporal_rep.size())
            temporal_representations.append(temporal_rep)
        
        concatenated_temporal_rep = torch.cat(temporal_representations, dim=-1)
        #print("concatendated_temporal_rep", concatenated_temporal_rep.size())
        output = self.feedforward(concatenated_temporal_rep)
        return output
    
    
class ModalitySpecificAttentionFusion(nn.Module):
    def __init__(self, input_dim):
        super(ModalitySpecificAttentionFusion, self).__init__()
        
        self.W_prime = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(2)])
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, representations):
        #print("ModalitySpecificAttentionFusion:")
        attention_weights = []
        for alpha in range(2):
            #print("representations ", representations[alpha].size())
            W_prime_alpha = self.W_prime[alpha](representations[alpha])
            #print("W_prime_alpha ", W_prime_alpha.size())
            attention_weights.append(W_prime_alpha)
            
        sum_W_prime = torch.stack(attention_weights, dim=0).sum(dim=0)
        #print("sum_W_prime ", sum_W_prime.size())
        
        normalized_attention_weights = self.softmax(sum_W_prime)
        
        attention_weights = {"video_self_att": normalized_attention_weights[:, :, 0].unsqueeze(-1).expand(-1, -1, 768),
                            "audio_self_att": normalized_attention_weights[:, :, 1].unsqueeze(-1).expand(-1, -1, 768)}
        #print("normalized_attention_weights ", normalized_attention_weights.size())
        #print("normalized_attention_weights[:, :, 0]", normalized_attention_weights[:, :, 0].size())
        #print("representations[0].unsqueeze(-1)", representations[0].size())
        #print("normalized_attention_weights[:, :, 1]", normalized_attention_weights[:, :, 1].size())
        #print("representations[1].unsqueeze(-1)", representations[1].size())
        #print("normalized_attention_weights[:, :, 2]", normalized_attention_weights[:, :, 2].unsqueeze(-1).expand(-1, -1, 768).size())
        #print("representations[2].unsqueeze(-1)", representations[2].size())
        
        #fused_representation = torch.sum(torch.cat([normalized_attention_weights[:, :, alpha] * representations[alpha].unsqueeze(-1) for alpha in range(3)], dim=-1),dim=-1)

        fused_representation = [normalized_attention_weights[:, :, alpha].unsqueeze(-1).expand(-1, -1, 768) * representations[alpha] for alpha in range(2)]
        
        final_fused_representation = fused_representation[0] + fused_representation[1]
        
        #print("final_fused_representation ", final_fused_representation.size())
        
        return final_fused_representation, attention_weights
    
    
class TemporalEnsembleWithFusion(nn.Module):
    def __init__(self, input_dim):
        super(TemporalEnsembleWithFusion, self).__init__()
        
        # Feed forward layer for fusion
        self.feed_forward_fusion = nn.Sequential(
            nn.Linear(input_dim, input_dim),  # Adjust the dimensions as needed
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)  # Adjust the dimensions as needed
        )
        
    def forward(self, Z, Z_fused):
        #print("TemporalEnsembleWithFusion:")
        Z_fused_ff = self.feed_forward_fusion(Z_fused)
        #print("Z ", Z.size())
        #print("Z_fused_ff ", Z_fused_ff.size())
        combined_representation = Z + Z_fused_ff
        #print("combined_representation ", combined_representation.size())
        
        return combined_representation # Pass this into final MLPs

class MultimodalModel(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, num_layers, num_heads, max_len, dropout, movement):
        super(MultimodalModel, self).__init__()
        
        self.movement = movement
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        
        self.video_conv1d_pe = TemporalConvolutionalLayer(embedding_dim[0], embedding_dim[0], max_len, dropout)
        self.audio_conv1d_pe = TemporalConvolutionalLayer(embedding_dim[0], embedding_dim[0], max_len, dropout)
        #self.text_conv1d_pe = TemporalConvolutionalLayer(embedding_dim[0], embedding_dim[0], max_len, dropout)
        
        self.multimodal_cross_transformer = MultimodalCrossTransformer(embedding_dim[0], num_heads)
        
        self.temporal_ensemble = TemporalEnsemble(num_layers, embedding_dim[0])
        
        self.modality_specific_self_attention = ModalitySpecificAttentionFusion(embedding_dim[0])
        
        self.temporal_ensemble_with_fusion = TemporalEnsembleWithFusion(embedding_dim[0])
        
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
                nn.Sigmoid()
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
        #print("Video shape:", video.shape)
        #print("audio shape:", audio.shape)
        #print("text shape:", text.shape)
        #print("mask shape:", mask.shape)
        #print("sub_clip shape", subclip_mask.shape)
        video_conv1d_pe = self.video_conv1d_pe(video, subclip_mask)
        audio_conv1d_pe = self.audio_conv1d_pe(audio, mask)
        #text_conv1d_pe = self.text_conv1d_pe(text, mask)
        
        #hidden_states, attention_scores_cross_transformer = self.multimodal_cross_transformer(video_conv1d_pe, audio_conv1d_pe, text_conv1d_pe)
        hidden_states, attention_scores_cross_transformer = self.multimodal_cross_transformer(video_conv1d_pe, audio_conv1d_pe)
        
        concatenated_temporal_rep = self.temporal_ensemble(hidden_states)
        
        fused_rep, attention_scores_self_att = self.modality_specific_self_attention([video_conv1d_pe, audio_conv1d_pe])
        
        combined_representation = self.temporal_ensemble_with_fusion(concatenated_temporal_rep, fused_rep)
        
        #combined_representation = combined_representation.view(combined_representation.shape[0], -1)
        
        #print(combined_representation.size())
        #print("Mask ", mask.size())
        #print("subclip Mask ", subclip_mask.size())
        
        padded_flat_input = torch.zeros(combined_representation.size(0), self.max_len * combined_representation.size(2), device=combined_representation.device)
        #print(padded_flat_input.size())
        padded_flat_input[:, :combined_representation.size(1) * combined_representation.size(2)] = (combined_representation * mask.unsqueeze(-1)).view(combined_representation.size(0), -1)
        #print(padded_flat_input.size())
        
        # MLPs for Asset Prices
        if not self.movement:
            index_large_output = self.asset_price_volatility_mlp_index_large(padded_flat_input)
            #print(index_large_output.size())
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
        
        return torch.cat([index_large_output, index_small_output, gold_output, dollar_output, ten_y_bond_output, three_m_bond_output],dim=1), attention_scores_cross_transformer, attention_scores_self_att