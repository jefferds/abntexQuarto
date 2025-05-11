# abntexQuarto

Este projeto busca somar a class abntex2 (<https://www.abntex.net.br/>) e o sistema de publicação científica e técnica Quarto.org. (<https://quarto.org/>).

### Utilização

Para utilização é necessário a instalação do sistema Quarto no site oficial. (<https://quarto.org/>).

Após a instalação do sistema, é necessário a instalação do Miktex (<https://miktex.org/download>) e do abntex2 (<https://www.abntex.net.br/>) para a geração do PDF.

O arquivo principal do projeto a ser renderizado (render PDF) é o `abntexQuarto.qmd`.

Para alterar dados como nome do autor, título, data e outras informações, basta editar o arquivo `_variables.yml`.

#### Linguagem de programação

Para a parte da programação e scripts utilizo a linguagem de programação Julia (<https://julialang.org/>).

#### Observações

1. Por alguma razão o sistema Quarto algumas vezes não consegue sincronizar arquivos em pastas como as do Google Drive. Manter seu projeto em uma pasta local (ex: C:\Users\SeuUsuario\Documents\) pode ajudar a evitar problemas de sincronização.
2. Algumas vezes o problema sincronização do Quarto pode ser resolvido deletando os arquivos temporários gerados pelo latex e a apagando a pasta `.quarto` (faça backup do seus arquivos antes de usar delete!).