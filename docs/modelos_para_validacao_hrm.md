# Modelos para validação

## 1. Modelos atuais do repositório

| Modelo | Arquitetura |
|---|---|
| ManometryCNN | CNN própria |
| ResNet18 | CNN residual |
| EfficientNet-B0 | CNN eficiente |
| ConvNeXt-Tiny | CNN moderna |
| DenseNet201 | CNN com conexões densas |
| Inception-v3 | CNN multiescala |
| ViT-B/16 | Vision Transformer |
| Wang CVP-GAT + ResNet18 | CNN com grafo de atenção |
| Wang CVP-GAT + DenseNet201 | DenseNet com grafo de atenção |

## 2. Novos modelos para imagens

### EfficientViT-M2

Transformer visual eficiente que reduz o custo de memória e a redundância da atenção. Será a alternativa principal ao ViT-B/16.

### MobileViT-v2

Modelo híbrido que combina convoluções para padrões locais e Transformer para relações globais. Será uma segunda referência de Transformer leve.

### DINOv2 ViT-S/14

Transformer visual pré-treinado por aprendizado auto-supervisionado. Será usado como extrator de características e também com fine-tuning.

## 3. Novos modelos para sinais brutos

### 1D-CNN

CNN unidimensional que aplica convoluções diretamente ao longo do tempo. Será o baseline neural temporal mais simples.

### ResNet1D-18

Adaptação da ResNet18 para séries temporais, utilizando blocos residuais e convoluções unidimensionais.

### InceptionTime

CNN temporal multiescala que usa diferentes tamanhos de kernel em paralelo para capturar padrões de curta e longa duração.

### TCN

Rede convolucional temporal com convoluções dilatadas, permitindo analisar sequências longas com processamento paralelo.

### BiLSTM

Rede recorrente bidirecional para modelar dependências temporais nos dois sentidos da sequência.

### BiGRU

Rede recorrente bidirecional semelhante à BiLSTM, mas com menos parâmetros e menor custo computacional.

### MiniROCKET

Transformação temporal baseada em convoluções quase determinísticas, seguida de um classificador linear. Será o principal baseline temporal de baixo custo.

### Transformer Encoder 1D

Transformer convencional aplicado diretamente aos tokens temporais. Será o controle para os Transformers temporais especializados.

### PatchTST

Transformer que divide a série em segmentos temporais e trata cada segmento como um token. Será o principal Transformer eficiente para sinais.

### MOMENT

Foundation model pré-treinado em diferentes conjuntos de séries temporais. Será validado como extrator de características e com fine-tuning.

### Wang Temporal-GAT

Extensão proposta do Wang CVP-GAT. Um encoder temporal extrai características dos sinais e um grafo de atenção modela as relações entre sensores ou regiões.

## 4. Novos modelos tabulares

### Regressão logística

Modelo linear de referência.

### Random Forest

Ensemble de árvores de decisão.

### XGBoost

Modelo de gradient boosting baseado em árvores.

### CatBoost

Modelo de gradient boosting voltado a dados tabulares.

### TabPFN-2.5

Foundation model tabular pré-treinado em tarefas sintéticas e baseado em aprendizado em contexto.

### TuneTables

Adaptação eficiente do TabPFN que aprende um contexto compacto, mantendo a maior parte do modelo congelada.

### DNNR

Modelo de regressão baseado em vizinhos, gradientes locais e aproximação de Taylor. Será validado somente em saídas contínuas.

## 5. Novos modelos multimodais

### Média de probabilidades

Combina as predições independentes dos modelos de imagem e sinal pela média.

### Concatenação + MLP

Concatena as representações da imagem e do sinal e utiliza uma rede densa para produzir a saída.

### Gated Fusion

Aprende pesos diferentes para controlar a contribuição de cada modalidade.

### Cross-Attention

Permite interação direta entre os tokens extraídos da imagem e os tokens extraídos do sinal.

### HRM-Bind

Modelo proposto com:

- EfficientViT ou DINOv2 como encoder de imagem;
- PatchTST ou MOMENT como encoder temporal;
- alinhamento contrastivo entre as duas representações;
- fusão por cross-attention;
- saída conjunta baseada nas duas modalidades.

## 6. Modelos prioritários

| Modalidade | Modelos |
|---|---|
| Imagem | todos os modelos atuais, EfficientViT-M2 e DINOv2-S |
| Sinal | 1D-CNN, ResNet1D-18, InceptionTime, MiniROCKET, PatchTST, MOMENT e Wang Temporal-GAT |
| Tabular | Regressão logística, Random Forest, XGBoost, CatBoost, TabPFN-2.5, TuneTables e DNNR |
| Multimodal | Média, Concatenação + MLP, Gated Fusion, Cross-Attention e HRM-Bind |

## 7. Modelos secundários

- MobileViT-v2;
- TCN;
- BiLSTM;
- BiGRU;
- Transformer Encoder 1D.
