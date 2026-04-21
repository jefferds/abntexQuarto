# Product Requirements Document (PRD)

## 1. Visão Geral
**Nome do Produto:** Template Acadêmico Quarto + ABNTeX2
**Objetivo:** Fornecer um template padronizado, automatizado e fácil de usar para a criação de trabalhos acadêmicos (TCCs, Monografias, Dissertações, Teses e Relatórios) no Brasil, aderindo estritamente às normas da ABNT utilizando o ecossistema Quarto e a classe tipográfica `abntex2` (LaTeX).

## 2. Público-Alvo
- Estudantes de Graduação (TCC / TG / Relatórios).
- Alunos de Pós-Graduação (Mestrado e Doutorado).
- Pesquisadores e Professores que buscam um framework reprodutível e robusto.

## 3. Casos de Uso
1. **Elaboração de Monografias e TCCs:** O usuário clona o repositório, ajusta os metadados em `_variables.yml` e foca apenas em escrever o conteúdo em arquivos Markdown (`.qmd`) dentro das pastas estruturadas.
2. **Trabalhos de Pós-Graduação:** Geração do documento com elementos complexos (dedicatória, epígrafe, folha de rosto e ficha catalográfica) já previamente alocados na estrutura.
3. **Escrita Mista:** Código em R/Python (com Quarto) interagindo diretamente com os blocos de texto acadêmico e normatização ABNT nativa.

## 4. Requisitos Funcionais (Funcionalidades)
- **Estrutura Modularizada:** O projeto deve estar dividido em pastas claras: `pretextuais/`, `textuais/` e `postestuais/` para fácil localização.
- **Configuração Centralizada:** Um arquivo (ex: `_variables.yml`) deve controlar todos os metadados bibliográficos do autor e da obra (nome, título, orientador, banca, etc.).
- **Normas ABNT (PDF):** O documento de saída primário (`.pdf`) deve herdar o `preamble.tex` utilizando a classe `abntex2` no Pandoc/XeLaTeX e usar o sistema `abntex2cite` ou CSL correspondente.
- **Ambiente Quarto:** Deve suportar compilação com o Quarto na versão mais recente, abstraindo a complexidade do LaTeX (sem o usuário precisar lidar com preâmbulos complexos).

## 5. Requisitos Não Funcionais
- **Manutenibilidade:** O código dos templates LaTeX subjacentes (`src/`) deve ser limpo e comentado.
- **Performance:** A compilação PDF deve ser ágil (cache ativo pelas configurações do Quarto).
- **Usabilidade:** Qualquer usuário com conhecimento básico de Markdown e Quarto deve conseguir compilar seu TCC copiando e colando textos limitando o contato com código LaTeX puro.

## 6. Fora de Escopo
- Modelos HTML interativos (foco primário da ABNT é impressão e leitura de PDFs).
- Formatação fora do padrão ABNT (APA, Vancouver, etc.).