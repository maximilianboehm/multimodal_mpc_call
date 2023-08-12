import torch
import torch.nn as nn
import math
import torch.nn.functional as F

"""
class ConvolutionalPositionalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, max_len):
        super(ConvolutionalPositionalEncoder, self).__init__()
        
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size)
        self.positional_encoding = nn.Embedding(max_len, hidden_dim) # maybe change to sinusoidalpositionalencoding
        
    def forward(self, x, mask):
        conv_output = self.conv(x)
        positional_encodings = self.positional_encoding(torch.arange(x.size(1), device=x.device))
        positional_encodings = positional_encodings.unsqueeze(0).expand(x.size(0), -1, -1)
        combined = conv_output + positional_encodings
        
        if mask is not None:
            combined = combined * mask.unsqueeze(-1)
        
        return combined
"""
class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_len, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, input_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_dim, 2).float()* (-math.log(10000.0) / input_dim))
        
        pe[:, 0::2] = torch.sin(position * di_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register.buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dopout(x)
    
class TemporalConvolutionalLayer(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, max_len, dropout):
        super(TemporalConvolutionalLayer, self).__init__()
        self.conv1d = nn. Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=kernel_size)
        self.pos_encoding = PositionalEncoding(input_dim, max_len, dropout)
        
    def forward(self, x):
        x = self.conv1d(x)
        x = self.pos_encoding(x)
        return x
        
"""
class CrossModalTransformer(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads, dropout):
        super(CrossModalTransformer, self).__init__()
        
        # Transformer from Video to Audio and Text to Audio
        self.transformer_VA = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers,
                                            num_decoder_layers=num_layers, dim_feedforward=hidden_dim*4,
                                            dropout=dropout)
        self.transformer_TA = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers,
                                            num_decoder_layers=num_layers, dim_feedforward=hidden_dim*4,
                                            dropout=dropout)
        
        #Transformer from Text to Video and Audio to Video
        self.transformer_TV = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers,
                                            num_decoder_layers=num_layers, dim_feedforward=hidden_dim*4,
                                            dropout=dropout)
        self.transformer_AV = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers,
                                            num_decoder_layers=num_layers, dim_feedforward=hidden_dim*4,
                                            dropout=dropout)
        
        # Transformer from Audio to Text and Video to Text
        self.transformer_AT = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers,
                                            num_decoder_layers=num_layers, dim_feedforward=hidden_dim*4,
                                            dropout=dropout)
        self.transformer_VT = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers,
                                            num_decoder_layers=num_layers, dim_feedforward=hidden_dim*4,
                                            dropout=dropout)
        
        def forward(self, video, audio, text, mask):
            # Cross Transformer for Audio
            cross_VA = self.transformer_VA(video, audio, tgt_mask=mask)
            cross_TA = self.transformer_TA(text, audio, tgt_mask=mask)
            
            # Cross Transformer for Video
            cross_AV = self.transformer_AV(audio, video, tgt_mask=mask)
            cross_TV = self.transformer_TV(text, video, tgt_mask=mask)
            
            # Cross Transformer for Text
            cross_VT = self.transformer_VT(video, text, tgt_mask=mask)
            cross_AT = self.transformer_AT(audio, text, tgt_mask=mask)
            
            return cross_VA, cross_TA, cross_AV, cross_TV, cross_VT, cross_AT
"""
class CrossModalAttention(nn.Module):
    def __init__(self, input_dim):
        super(CrossModalAttention, self).__init__()
        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)
        self.W_ff = nn.Linear(input_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, Zi_1_alpha_beta, Zi_1_alpha):
        Z_i_alpha_beta = self.layer_norm(Zi_1_alpha_beta)
        Z_i_alpha = self.layer_norm(Zi_1_alpha)
        
        q = self.W_q(Z_i_alpha_beta)
        k = self.W_k(Z_i_alpha)
        v = self.W_v(Z_i_alpha)
        
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        attention_weights = torch.nn.functional.softmax(attention_scores / math.sqrt(Z_i_alpha.size(-1)), dim=-1)

        attended_representation = torch.matmul(attention_weights, v)
        corss_modal_adaption = attended_representation + Z_i_alpha_beta
        
        z_i_alpha_to_beta_intermediate = self.layer_norm(cross_modal_adaption)
        
        z_i_alpha_to_beta_ff = self.W_ff(z_i_alpha_to_beta_intermediate)
        z_i_alpha_to_beta_final = self.layer_norm(z_i_alpha_to_beta_intermediate + z_i_alpha_to_beta_ff)
        
        return z_i_alpha_to_beta_final
        
class MultimodalCrossTransformer(nn.Module):
    def __init__(self, input_dim):
        super(MultimodalCrossTransformer, self).__init__()
        self.audio_to_video_attention = CrossModalAttention(input_dim)
        self.text_to_video_attention = CrossModalAttention(input_dim)
        
        self.video_to_audio_attention = CrossModalAttention(input_dim)
        self.text_to_audio_attention = CrossModalAttention(input_dim)
        
        self.video_to_text_attention = CrossModalAttention(input_dim)
        self.audio_to_text_attention = CrossModalAttention(input_dim)
        
    def forward(self, video, audio, text):
        # Video Modality
        audio_to_video = self.audio_to_video_attention(audio, video)
        text_to_video = self.text_to_video_attention(text, video)
        video_attention = torch.cat([audio_to_video, text_to_video])
        
        # Audio Modality
        video_to_audio = self.video_to_audio_attention(video, audio)
        text_to_audio = self.text_to_audio_attention(text, audio)
        audio_attention = torch.cat([video_to_audio, text_to_audio])
        
        # Text Modality
        video_to_text = self.video_to_text_attention(video, text)
        audio_to_text = self.audio_to_text_attention(audio, text)
        text_attention = torch.cat([video_to_text, audio_to_text])
        
        return video_attention, audio_attention, text_attention
        
        
class TemporalEnsemble(nn.Module):
    def __init__(self, num_layers, input_dim, output_dim):
        super(TemporalEnsemble, self).__init__()
        
        self.transformers = nn.ModulesList([nn.Transformer(d_model=input_dim, nhead=3, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dim_feedforward=input_dim*4) for _ in range(3)])
        self.feedforward = nn.Linear(input_dim, input_dim) # change to simple MLP with Relu?
        
    def forward(self, hidden_states_list):
        num_modalities = len(hidden_states_list)
        temporal_representation = []
        
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
        
        self.W_prime = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_modalities)])
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, representation):
        attention_weights = []
        for alpha in range(3):
            W_prime_alpha = self.W_prime[alpha](representations[alpha])
            attention_weights.append(W_prime_alpha)
            
        sum_W_prime = torch.stack(attention_weights, dim=0).sum(dim=0)
        
        normalized_attention_weights = self.softmax(sum_W_prime)
        
        fused_representation = sum([normalized_attention_weight.unsqueeze(-1) * representation for normalized_attention_weight, representation in zip(normalized_attention_weights, representations)])
        
        return fused_representation
    
class TemporalEnsembleWithFusion(nn.Module):
    def __init__(self, input_dim):
        super(TemporalEnsembleWithFusion, self).__init__()
        
        # Feed forward layer for fusion
        self.feed_forward_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # Adjust the dimensions as needed
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)  # Adjust the dimensions as needed
        )
        
    def forward(self, Z, Z_fused):
        Z_fused_ff = self.feed_forward_fusion(Z_fused)
        combined_representation = Z + Z_fused_ff
        
        return combined_representation # Pass this into final MLPs
            
        
class MultimodalModel(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, num_layers, num_heads, max_len, dropout, movement):
        super(MultimodalModel, self).__init__()
        
        # Convolutional Positional Encoders
        #self.video_encoder = ConvolutionalPositionalEncoder(embedding_dim, hidden_dim, kernel_size=3, max_len=max_len)
        #self.audio_encoder = ConvolutionalPositionalEncoder(embedding_dim, hidden_dim, kernel_size=3, max_len=max_len)
        #self.text_encoder = ConvolutionalPositionalEncoder(embedding_dim, hidden_dim, kernel_size=3, max_len=max_len)
        self.video_encoder = TemporalConvolutionalLayer(embedding_dim[0], embedding_dim[0], kernel_size=3, max_len, dropout)
        self.audio_encoder =TemporalConvolutionalLayer(embedding_dim[1], embedding_dim[1], kernel_size=3, max_len, dropout)
        self.text_encoder = TemporalConvolutionalLayer(embedding_dim[2], embedding_dim[2], kernel_size=3, max_len, dropout)
        
        # Cross Modal Transformers
        self.cross_modal_transformer = CrossModalTransformer(hidden_dim, num_layers, num_heads, dropout)
        
        # Attention Fusion
        self.attention_fusion = nn.Transformer(d_model=hidden_dim*3, nhead=num_heads, num_encoder_layers=num_layers,
                                              num_decoder_layers=num_layers, dim_feedforward=hidden_dim * 12,
                                              dropout=dropout)
        
        # maybe add more layers according to num layers
        if not movement:
            # Asset price volatility prediction
            self.asset_price_volatility_mlp_index_large = nn.Sequential(
                nn.Linear(mlp_input_dim, mlp_hidden_dim),
                nn.Linear(mlp_hidden_dim, mlp_output_dim)
            )
            self.asset_price_volatility_mlp_index_small = nn.Sequential(
                nn.Linear(mlp_input_dim, mlp_hidden_dim),
                nn.Linear(mlp_hidden_dim, mlp_output_dim)
            )
            self.asset_price_volatility_mlp_gold = nn.Sequential(
                nn.Linear(mlp_input_dim, mlp_hidden_dim),
                nn.Linear(mlp_hidden_dim, mlp_output_dim)
            )
            self.asset_price_volatility_mlp_dollar = nn.Sequential(
                nn.Linear(mlp_input_dim, mlp_hidden_dim),
                nn.Linear(mlp_hidden_dim, mlp_output_dim)
            )
            self.asset_price_volatility_mlp_10_y_bond = nn.Sequential(
                nn.Linear(mlp_input_dim, mlp_hidden_dim),
                nn.Linear(mlp_hidden_dim, mlp_output_dim)
            )
            self.asset_price_volatility_mlp_3_m_bond = nn.Sequential(
                nn.Linear(mlp_input_dim, mlp_hidden_dim),
                nn.Linear(mlp_hidden_dim, mlp_output_dim)
            )
        
        else:
            # Asset price movement prediction
            self.asset_price_movement_mlp_index_large = nn.Sequential(
                nn.Linear(mlp_input_dim, mlp_hidden_dim),
                nn.Sigmoid(), 
                nn.Linear(mlp_hidden_dim, mlp_output_dim)
            )
            self.asset_price_movement_mlp_index_small = nn.Sequential(
                nn.Linear(mlp_input_dim, mlp_hidden_dim),
                nn.Sigmoid(), 
                nn.Linear(mlp_hidden_dim, mlp_output_dim)
            )
            self.asset_price_movement_mlp_gold = nn.Sequential(
                nn.Linear(mlp_input_dim, mlp_hidden_dim),
                nn.Sigmoid(), 
                nn.Linear(mlp_hidden_dim, mlp_output_dim)
            )
            self.asset_price_movement_mlp_dollar = nn.Sequential(
                nn.Linear(mlp_input_dim, mlp_hidden_dim),
                nn.Sigmoid(), 
                nn.Linear(mlp_hidden_dim, mlp_output_dim)
            )
            self.asset_price_movement_mlp_10_y_bond = nn.Sequential(
                nn.Linear(mlp_input_dim, mlp_hidden_dim),
                nn.Sigmoid(), 
                nn.Linear(mlp_hidden_dim, mlp_output_dim)
            )
            self.asset_price_movement_mlp_3_m_bond = nn.Sequential(
                nn.Linear(mlp_input_dim, mlp_hidden_dim),
                nn.Sigmoid(), 
                nn.Linear(mlp_hidden_dim, mlp_output_dim)
            )

    def forward(self, video, audio, text, mask, subclip_mask):
        # Apply Convolutional Positional Encoding
        video_encoded = self.video_encoder(video, mask)
        audio_encoded = self.audio_encoder(audio, mask)
        text_encoded = self.text_encoder(text, mask)
        
        # Perform Cross Modal Transformation
        cross_VA, cross_TA, cross_AV, cross_TV, cross_VT, cross_AT = self.cross_modal_transformer(video, audio, text, mask)
        
        # Attention Fusion
        combined = torch.cat((cross_VA, cross_TA, cross_AV, cross_TV, cross_VT, cross_AT), dim=2)
        final_output = self.attention_fusion(combined, combined, tgt_mask=mask)
        
        # MLPs for Asset Prices
        if not movement:
            index_large_output = self.asset_price_volatility_mlp_index_large
            index_small_output = self.asset_price_volatility_mlp_index_small
            gold_output = self.asset_price_volatility_mlp_gold
            dollar_output = self.asset_price_volatility_mlp_dollar
            ten_y_bond_output = self.asset_price_volatility_mlp_10_y_bond
            three_m_bond_output = self.asset_price_volatility_mlp_3_m_bond
        else:
            index_large_output = self.asset_price_movement_mlp_index_large
            index_small_output = self.asset_price_movement_mlp_index_small
            gold_output = self.asset_price_movement_mlp_gold
            dollar_output = self.asset_price_movement_mlp_dollar
            ten_y_bond_output = self.asset_price_movement_mlp_10_y_bond
            three_m_bond_output = self.asset_price_movement_mlp_3_m_bond
        
        return index_large_output, index_small_output, gold_output, dollar_output, ten_y_bond_output, three_m_bond_output