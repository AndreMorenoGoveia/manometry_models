# Plano de pesquisa: classificação multimodal de manometria esofágica

**Versão:** 20 de julho de 2026  
**Modalidades:** imagens topográficas de HRM e sinais brutos multicanais (pressão × tempo)  
**Unidade de análise principal:** exame/paciente, não imagem ou deglutição isolada

## 1. Pergunta e hipóteses

### Pergunta principal

Qual representação — imagem topográfica, sinal bruto ou fusão das duas modalidades — produz o diagnóstico mais correto, calibrado, robusto e eficiente de distúrbios motores esofágicos?

### Hipóteses

1. Modelos nativos de séries temporais preservarão informação quantitativa perdida na renderização da imagem.
2. Modelos pré-treinados superarão modelos treinados do zero no cenário de poucos pacientes.
3. Um Transformer eficiente terá relação desempenho/custo melhor que o ViT-B/16 atual.
4. A fusão de imagem e sinal do mesmo exame superará cada modalidade isolada.
5. Alinhamento contrastivo entre modalidades permitirá inferência robusta mesmo quando uma delas estiver ausente.
6. Descritores clínicos tabulares e modelos transparentes serão competitivos e ajudarão a explicar erros.

## 2. Desfechos

### Primário

Classificação multiclasse no nível do exame. Inicialmente, manter as seis classes existentes:

- acalasia tipo II (renomear o rótulo técnico atual `Bradycardia_type_II`);
- DES;
- EGJOO;
- IEM;
- esôfago hipercontrátil/Jackhammer;
- normal.

Os rótulos devem ser auditados por especialistas e alinhados à Chicago Classification v4.0 antes do congelamento do banco.

### Secundários

- classificação hierárquica: normal × anormal; depois subtipo;
- classificação por deglutição, agregada para o exame;
- regressão de IRP, DCI, latência distal e outros índices disponíveis;
- detecção de exame fora de distribuição/qualidade inadequada;
- concordância com especialistas e desempenho assistido por IA.

## 3. Preparação dos dados

### 3.1 Estrutura canônica

Cada exemplo deve possuir:

- `patient_id`, `exam_id`, centro, equipamento e data;
- matriz bruta `X ∈ R^(C×T)`, com máscara para canais/instantes ausentes;
- imagem topográfica gerada a partir do mesmo trecho;
- deglutições e regiões anatômicas anotadas, quando disponíveis;
- variáveis clínicas/descritores manométricos;
- rótulo e versão do consenso diagnóstico.

Guardar sinais em unidade física, taxa de amostragem e resolução originais. A imagem deve ser regenerável com mapa de cores, escala de pressão e recorte documentados.

### 3.2 Controle de vazamento

O repositório atual contém **12 arquivos exatamente duplicados entre treino e validação/teste**. Além disso, validação e teste têm a mesma contagem por classe (461 imagens cada). Os resultados atuais devem ser considerados exploratórios até refazer os splits.

Regras:

- dividir por paciente e exame antes de qualquer segmentação ou aumento;
- manter todas as deglutições, imagens e versões aumentadas de um paciente no mesmo fold;
- deduplicar por hash exato e por hash perceptual;
- detectar imagens derivadas do mesmo original;
- separar um teste externo por centro/equipamento, se houver;
- congelar o teste antes da seleção de arquitetura e hiperparâmetros.

### 3.3 Divisão recomendada

- Desenvolvimento: validação cruzada estratificada e agrupada por paciente, 5 folds.
- Teste interno: 15–20% dos pacientes, intocado.
- Teste externo: outro centro, equipamento ou período temporal.
- Repetir a divisão com 3 sementes quando o número de pacientes permitir.

Se houver poucos pacientes por classe, usar nested cross-validation agrupada e reportar intervalos de confiança, sem um único split otimista.

### 3.4 Pré-processamento

**Imagem**

- remover identificadores, molduras, texto e elementos que revelem equipamento/diagnóstico;
- preservar proporção pressão/tempo; evitar recorte que elimine EGJ;
- normalização ImageNet somente para pesos correspondentes;
- aumentos clinicamente plausíveis e iguais entre métodos: pequenas translações/escala, leve ruído e variação fotométrica;
- não usar inversão temporal, flip vertical ou alterações que mudem anatomia/fisiologia.

**Sinal bruto**

- reamostrar para uma frequência comum e conservar a frequência original como metadado;
- normalizar por exame/canal de forma definida apenas no treino;
- máscara explícita para sensores defeituosos;
- janelar por deglutição e também avaliar o protocolo completo;
- aumentos plausíveis: jitter pequeno, amplitude scaling, time masking e channel dropout;
- não usar warping ou permutação que altere os critérios fisiológicos sem validação clínica.

## 4. Princípio de comparação justa

“Os mesmos modelos” será operacionalizado de três formas:

1. **Entrada equivalente:** o mesmo sinal alimenta o modelo como matriz bruta e como imagem renderizada.
2. **Famílias pareadas:** CNN, recorrente e Transformer possuem uma versão 1D e uma versão 2D com orçamento aproximado de parâmetros.
3. **Backbone realmente compartilhado:** imagem e sinal são convertidos em tokens, recebem embeddings de modalidade e passam pelo mesmo Transformer.

Não se deve forçar uma CNN 2D diretamente sobre uma sequência 1D sem declarar a transformação. Modelos nativos de cada modalidade serão necessários para uma comparação cientificamente válida.

## 5. Baselines

### 5.1 Controles sem deep learning

Aplicar sobre descritores manométricos clínicos e estatísticos extraídos **sem usar o teste**:

| ID | Modelo | Papel |
|---|---|---|
| C0 | regra clínica/Chicago v4.0 | referência interpretável |
| C1 | regressão logística multinomial | piso linear |
| C2 | k-NN | baseline local |
| C3 | SVM-RBF | baseline clássico não linear |
| C4 | Random Forest | ensemble clássico |
| C5 | XGBoost | boosting |
| C6 | CatBoost | boosting robusto para banco pequeno/misto |
| C7 | MLP tabular | rede densa simples |
| C8 | TabPFN v2 ou TabPFN-2.5 | foundation model tabular |
| C9 | TuneTables | ajuste eficiente de contexto do TabPFN |
| C10 | DNNR | **somente regressão** de IRP/DCI/latência; opcional one-vs-rest exploratório |

TuneTables deve ser comparado com o TabPFN sem ajuste, CatBoost e XGBoost. A contribuição a testar é a compressão do conjunto em um contexto aprendido com poucos parâmetros ajustáveis. TabPFN-2.5 deve ser registrado com versão, execução local/cloud e licença; não misturar seus resultados com os do TabPFN original.

DNNR deve incluir as ablações do artigo: KNN, KNN com feature scaling, DNNR de primeira ordem e, se numericamente estável, DNNR de segunda ordem.

### 5.2 Todos os modelos já presentes no repositório — imagem

| ID | Modelo | Inicialização | Situação |
|---|---|---|---|
| I0 | `cnn` (ManometryCNN) | do zero | implementado |
| I1 | `resnet18` | ImageNet | implementado |
| I2 | `efficientnet_b0` | ImageNet | implementado |
| I3 | `convnext_tiny` | ImageNet | implementado |
| I4 | `densenet201` | ImageNet | implementado |
| I5 | `inception_v3` | ImageNet | implementado |
| I6 | `vit_base` (ViT-B/16, SWAG, 384 px) | ImageNet/SWAG | implementado |
| I7 | `wang_cvp_gat` + ResNet18 | ImageNet | implementado |
| I8 | `wang_cvp_gat_densenet201` | ImageNet | implementado |

Os nove devem ser executados novamente após a correção do split. A comparação Wang CVP-GAT deve incluir:

- ResNet18 sem GAT × ResNet18 + GAT;
- DenseNet201 sem GAT × DenseNet201 + GAT;
- ablação do número de nós, raio do grafo e correlação posicional;
- nós definidos por anatomia, quando houver landmarks, versus pooling uniforme.

### 5.3 Novos baselines de imagem

| ID | Modelo | Justificativa |
|---|---|---|
| I9 | EfficientViT-M2 ou M3 | Transformer eficiente principal; atenção em grupos em cascata e baixa latência |
| I10 | MobileViT-v2-1.0 | segundo Transformer móvel para confirmar que o resultado não depende de uma arquitetura |
| I11 | DINOv2 ViT-S/14 + linear probe/fine-tuning | representação visual auto-supervisionada forte |

**Escolha principal para substituir/complementar o ViT-B:** EfficientViT-M2. Ele deve ser comparado com ViT-B/16 em macro-F1, parâmetros, FLOPs, memória de pico, imagens/s e latência batch 1 no hardware-alvo. MobileViT-v2 é uma análise de sensibilidade, não precisa entrar na etapa mais cara se os recursos forem limitados.

Para DINOv2, avaliar primeiro encoder congelado + cabeça linear e depois fine-tuning parcial/LoRA. Isso separa qualidade da representação de capacidade de ajuste.

### 5.4 Baselines nativos de sinais brutos

| ID | Modelo | Papel |
|---|---|---|
| S0 | estatísticas + C1–C9 | ponte tabular e interpretável |
| S1 | 1D-CNN pareada à ManometryCNN | baseline simples |
| S2 | ResNet1D-18 | contraparte da ResNet18 |
| S3 | InceptionTime | CNN multiescala forte para classificação temporal |
| S4 | TCN | convolução causal/dilatada |
| S5 | BiLSTM | baseline recorrente |
| S6 | BiGRU | recorrente mais eficiente |
| S7 | MiniROCKET + Ridge/logística | baseline rápido e difícil de superar |
| S8 | Transformer Encoder 1D | Transformer treinado do zero |
| S9 | PatchTST-classifier | Transformer eficiente por patches |
| S10 | MOMENT-small/base + cabeça de classificação | foundation model de séries temporais |
| S11 | Wang temporal-GAT | canais/sensores como nós e arestas anatômicas |

O PatchTST é o Transformer eficiente principal para o sinal, pois o patching reduz o comprimento da atenção. O MOMENT testa transferência de um foundation model temporal. Para HRM multivariada, comparar:

- canais independentes com pesos compartilhados;
- mistura explícita entre canais;
- compressão multivariada/long-context;
- sinal por deglutição versus protocolo completo.

O Wang temporal-GAT é a extensão mais direta e clinicamente motivada do modelo do repositório: cada sensor/canal é um nó ordenado espacialmente; uma CNN/Transformer temporal produz atributos por nó; o GAT modela dependências locais e de longa distância.

### 5.5 Comparação pareada por família

| Família | Imagem | Sinal |
|---|---|---|
| simples | ManometryCNN | 1D-CNN |
| residual | ResNet18 | ResNet1D-18 |
| multiescala | Inception-v3 | InceptionTime |
| eficiente | EfficientNet-B0 | MiniROCKET/TCN |
| Transformer grande | ViT-B/16 | Transformer Encoder 1D |
| Transformer eficiente | EfficientViT-M2 | PatchTST |
| foundation model | DINOv2-S | MOMENT |
| grafo | Wang CVP-GAT | Wang temporal-GAT |

## 6. Proposta multimodal: HRM-Bind

### 6.1 Arquitetura

Propor um modelo de fusão inspirado no princípio do ImageBind, adaptado a HRM pareada:

```text
imagem ── EfficientViT/DINOv2 ── tokens visuais ─┐
                                                 ├─ espaço latente alinhado
sinal ─── PatchTST/MOMENT ─────── tokens temporais┘
                    │
descritores ─ TabPFN/MLP ─ tokens tabulares (opcional)
                    │
        cross-attention + gated fusion
                    │
     diagnóstico + índices + qualidade/OOD
```

Componentes:

- encoder visual eficiente: EfficientViT-M2; DINOv2-S como variante de maior capacidade;
- encoder temporal: PatchTST; MOMENT como variante pré-treinada;
- projetores pequenos para uma dimensão latente comum;
- perda contrastiva simétrica imagem–sinal usando pares do mesmo exame;
- fusão por cross-attention com gate por modalidade;
- cabeça multiclasse no nível da deglutição e agregador attention/MIL no nível do exame;
- cabeças auxiliares de regressão para IRP/DCI e de qualidade;
- modality dropout durante treino para tolerar modalidade ausente.

### 6.2 Etapas de treinamento

1. Treinar/ajustar cada encoder unimodal separadamente.
2. Pré-alinhar imagem e sinal pareados com perda contrastiva, sem usar rótulos.
3. Treinar a fusão com `L = Lclasse + λcLcontrastiva + λrLregressão + λqLqualidade`.
4. Fine-tuning conjunto com learning rates menores nos encoders.
5. Calibrar probabilidades somente no fold de validação.

### 6.3 Baselines de fusão obrigatórios

| ID | Fusão | Objetivo |
|---|---|---|
| M0 | média de probabilidades | late fusion sem parâmetros |
| M1 | concatenação de embeddings + MLP | baseline de fusão simples |
| M2 | gated late fusion | ponderar qualidade/disponibilidade |
| M3 | cross-attention | interações token a token |
| M4 | HRM-Bind contrastivo + cross-attention | proposta completa |
| M5 | M4 + descritores/TabPFN | fusão trimodal opcional |

### 6.4 Ablations multimodais

- imagem apenas, sinal apenas e ambos;
- sem perda contrastiva;
- sem pré-treinamento;
- encoder congelado × fine-tuning completo × LoRA/adapters;
- concatenação × gate × cross-attention;
- sem modality dropout;
- sem cabeças auxiliares;
- sem descritores tabulares;
- modalidade artificialmente corrompida ou ausente;
- alinhamento correto × pares embaralhados como controle negativo.

## 7. Protocolo experimental

### 7.1 Treinamento

- mesma partição agrupada para todos os modelos;
- busca de hiperparâmetros somente nos folds de desenvolvimento;
- orçamento comparável por família (número de tentativas e tempo);
- early stopping por macro-F1;
- class weights ou focal loss escolhidos apenas na validação;
- 3–5 sementes para redes neurais;
- registrar versão do código, seed, fold, pesos, transformações e ambiente;
- treinos mistos FP16/BF16 quando suportados, mas métricas finais em precisão estável.

Dois regimes devem ser reportados separadamente:

1. **Treino controlado:** mesma resolução, augmentations, scheduler e orçamento.
2. **Melhor receita por modelo:** configuração recomendada para cada arquitetura.

O primeiro mede arquitetura; o segundo mede o melhor desempenho alcançável.

### 7.2 Métricas preditivas

Primárias:

- macro-F1;
- balanced accuracy;
- macro-AUROC one-vs-rest.

Secundárias:

- acurácia e weighted-F1;
- sensibilidade, especificidade, PPV e NPV por classe;
- macro-AUPRC;
- matriz de confusão;
- top-2 accuracy para análise de diagnóstico diferencial;
- MAE/RMSE/R² e Bland–Altman para índices contínuos;
- ECE, Brier score e curvas de calibração;
- cobertura × risco para opção de abstenção.

Reportar IC 95% por bootstrap **agrupado no paciente**. Comparar classificadores pareados com bootstrap de diferenças e, quando adequado, McNemar; corrigir múltiplas comparações (Holm). Não selecionar o “vencedor” apenas por acurácia.

### 7.3 Eficiência

Para cada modelo:

- parâmetros treináveis e totais;
- FLOPs/MACs;
- memória máxima no treino e inferência;
- tempo de treino;
- latência batch 1 e throughput no mesmo hardware;
- tamanho do checkpoint;
- consumo energético, se mensurável.

Construir fronteira de Pareto macro-F1 × latência × memória. O Transformer “mais eficiente” será escolhido nessa fronteira, não pelo nome da arquitetura.

### 7.4 Robustez e generalização

- validação externa por centro/equipamento;
- leave-one-center/device-out;
- ruído, canais ausentes, mudança de taxa de amostragem e escala;
- variações de renderização/mapa de cores;
- subgrupos por sexo, idade e centro, quando ética e estatisticamente possíveis;
- desempenho por qualidade do exame;
- OOD: classe desconhecida e equipamento não visto;
- teste de modalidade ausente no multimodal.

### 7.5 Interpretabilidade

- imagem: Grad-CAM para CNNs e attention rollout/occlusion para Transformers;
- sinal: Integrated Gradients/occlusion por tempo e canal;
- tabular: SHAP/permutation importance e contexto aprendido do TuneTables;
- DNNR: vizinhos, gradientes locais e falhas por condicionamento;
- grafo: importância de nós/arestas;
- validação cega por especialistas: relevância anatômica/fisiológica, não apenas figuras ilustrativas.

## 8. Execução em fases

### Fase 0 — governança e auditoria

- corrigir nomenclatura e ontologia;
- reconstruir IDs e splits por paciente;
- deduplicar;
- documentar geração imagem–sinal;
- definir aprovação ética e política de anonimização.

**Gate:** nenhum treino definitivo antes de passar a auditoria de vazamento.

### Fase 1 — baselines mínimos

- C0–C6;
- I0–I8 (todos os modelos do repositório);
- S1, S2, S3, S7 e S9.

**Entrega:** tabela unimodal completa e estimativa de variância entre folds.

### Fase 2 — pré-treinamento e eficiência

- EfficientViT-M2, MobileViT-v2, DINOv2-S;
- Transformer 1D, MOMENT;
- medir Pareto de eficiência.

**Gate:** escolher no máximo dois encoders por modalidade para fusão.

### Fase 3 — ideias dos artigos e grafos

- descritores + TabPFN/TabPFN-2.5/TuneTables;
- DNNR para índices contínuos;
- Wang temporal-GAT e ablações anatômicas.

### Fase 4 — multimodal

- M0–M5;
- ablações;
- modalidade ausente e corrupção;
- calibração e abstenção.

### Fase 5 — validação externa e estudo clínico

- conjunto externo congelado;
- comparação com especialistas;
- leitor sem IA × com IA, se o desenho permitir;
- relatório conforme TRIPOD-AI/CONSORT-AI quando aplicável.

## 9. Conjunto mínimo publicável e conjunto completo

### Mínimo publicável

- clássicos: regressão logística, Random Forest, XGBoost/CatBoost;
- imagem: CNN, ResNet18, DenseNet201, ViT-B, EfficientViT, Wang CVP-GAT;
- sinal: 1D-CNN, ResNet1D, InceptionTime, MiniROCKET, PatchTST;
- multimodal: média, concatenação e HRM-Bind;
- split por paciente, 5 folds, teste externo, calibração e eficiência.

### Completo

Todos os IDs C0–C10, I0–I11, S0–S11 e M0–M5. Para controlar custo, usar successive halving: todos recebem orçamento curto; apenas candidatos no Pareto de validação avançam para múltiplas sementes e teste externo.

## 10. Critério de decisão

O modelo final não será necessariamente o de maior acurácia. Recomenda-se:

1. excluir modelos sem generalização externa ou mal calibrados;
2. formar o Pareto macro-F1 × latência × memória;
3. preferir o modelo mais simples dentro de 1 erro-padrão do melhor;
4. exigir sensibilidade mínima por classe clinicamente definida;
5. manter uma opção de abstenção para baixa confiança/OOD;
6. congelar modelo e limiar antes do teste externo.

## 11. Referências que fundamentam as adições

- Feuer et al. **TuneTables: Context Optimization for Scalable Prior-Data Fitted Networks**. NeurIPS 2024. Artigo fornecido: `2402.11137v3.pdf`.
- Prior Labs Team. **TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models**. 2026. Artigo fornecido: `2511.08667v2.pdf`.
- Nader, Sixt e Landgraf. **DNNR: Differential Nearest Neighbors Regression**. ICML 2022. Artigo fornecido: `nader22a.pdf`.
- Liu et al. **EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention**. https://arxiv.org/abs/2305.07027
- Nie et al. **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers**. https://arxiv.org/abs/2211.14730
- Goswami et al. **MOMENT: A Family of Open Time-series Foundation Models**. https://arxiv.org/abs/2402.03885
- Girdhar et al. **ImageBind: One Embedding Space To Bind Them All**. https://arxiv.org/abs/2305.05665
- Dempster et al. **MINIROCKET**. https://arxiv.org/abs/2012.08791
- Ismail Fawaz et al. **InceptionTime**. https://arxiv.org/abs/1909.04939
- Oquab et al. **DINOv2**. https://arxiv.org/abs/2304.07193

