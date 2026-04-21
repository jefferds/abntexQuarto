# System Specification (SEPEC)

## 1. Introdução da Arquitetura
O sistema será estruturado puramente a partir de parâmetros do **Quarto Publishing System** sobrelaçado à distribuição TeX/LaTeX contendo o pacote `abntex2`. Ele faz uso da diretiva `include` para a injeção modular da monografia e metadados lidos através de variáveis de projeto YAML.

## 2. Estrutura de Diretórios Proposta / Atual
A composição da árvore segue um padrão acadêmico direto:
```text
/
├── _quarto.yml          # Core do build. Define os perfis de build, engines (pdflatex/xelatex) e referências bibliográficas.
├── _variables.yml       # Fonte unitária de verdade para Metadados (Nome, Título, Banca, Folha de Aprovação).
├── abntexQuarto.qmd     # Documento "master" que orquestra a união das partes incluídas.
├── references.bib       # Banco de dados de referências no formato BibTeX.
├── src/
│   ├── preamble.tex     # Injeções de pacotes na classe abntex2 (fontes, babel, ajustes finos).
│   └── styles.css       # Se usado exportação para HTML/EPUB no futuro.
├── pretextuais/         # Capa, Folha de Rosto, Epígrafe, Abstract, Resumo, Sumário, Listas.
├── textuais/            # Capítulos modulares numéricos (cap1_*.qmd a capN_*.qmd).
└── postestuais/         # Apêndices, Anexos e Bibliografia final.
```

## 3. Fluxo de Compilação
1. O usuário invoca `quarto render abntexQuarto.qmd`.
2. O **Pandoc** engloba todos os `.qmd` do projeto lendo as instruções no YAML Header e o `_quarto.yml`.
3. Metadados do autor mapeados em `_variables.yml` são interpolados no ambiente LaTeX via lua-filters (se aplicável), ou no preâmbulo nativo estendido do Quarto.
4. O compilador (ex. `xelatex` ou `pdflatex`) embute os parágrafos convertidos de Markdown em chamadas macro do `abntex2`.
5. Como step de bibliografia, CSL da ABNT (`abnt2.csl`) ou chamadas do `abntex2cite.sty` interpretam as citações `@referencia` para dentro da norma correta e imprimem do .bib.
6. A saída `abntexQuarto.pdf` é ejetada com os sumários lógicos prontos.

## 4. Integrações de Core Técnicas
- **Bibliotecas TeX Essenciais:** `abntex2`, `microtype`, `fontspec`, `hyperref`. 
- **Suporte Quarto:** A ferramenta exige metadados estruturados. `template-partials` pode ser usado futuramente, mas a versão atual se foca no wrapper `.qmd` raiz e preâmbulo injetável em `include-in-header: src/preamble.tex`.
- **Estilos:** CSL da ABNT referenciado em `csl: abnt2.csl` presente na raiz do projeto.

## 5. Práticas de Desenvolvimento e Expansão
- Adição de Capítulos: Devem ser gerados em `textuais/` e importados no root (ou listados via diretivas do tipo `book`). Note que como estamos usando `qmd` tradicional, usaremos sintaxes Shortcodes `{{< include textuais/capX.qmd >}}` no arquivo principal `abntexQuarto.qmd` para construir o PDF monolítico.
- Imagens / Gráficos: Guardados sempre na pasta `img/` ou referenciados com caminho nativo do Quarto via chunks `knitr/jupyter`.