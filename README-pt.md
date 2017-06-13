# Projeto de Aprendizagem de Máquina 2017.1

Para configurar seu ambiente para rodar os experimentos, siga os passos abaixo a depender do seu Sistema Operacional.

## Linux

Os passos aqui apresentados foram realizados em uma máquina Ubuntu 16.04, mas dificilmente algo daqui não se aplicará a sua distribuição
preferida, a menos do gerenciador de pacotes.

Você precisa primeiro instalar o Python 2.7.12 ou superior e o gerenciador de pacotes Python, pip. Isso pode ser feito com o seguinte
commando de terminal:

```sh
$ sudo apt-get install python python-pip
```

Lembrando que para outras distribuições, você deve utilizar o seu próprio gerenciador de pacotes. Por exemplo, no Fedora, teria de ser
utilizado o `yum`no lugar de `apt-get`.

Uma vez que o python e o pip estão instalados, você deve instalar as dependências do projeto. Para isso, execute os seguintes comandos
no terminal:

```sh
$ sudo pip install -r requirements.txt
```

É importante que você execute esse comando na pasta principal do projeto, pois é onde se encontra o arquivo `requirements.txt`.

Por fim, você pode instalar o projeto em si:

```sh
$ sudo pip install -e .
```

Isso fará com que o projeto seja instalado em modo `symlink` o que significa que quaisquer modificações feitas no código-fonte serão
refletidas na solução instalada.

## Windows

Os passos aqui apresentados foram realizados em uma máquina com Windows 10, mas dificilmente algo daqui não se aplicará a sua
versão do Windows.

Você precisa primeiro instalar o Python 2.7.12 ou superior. No caso do Windows, o instalador padrão já vem com o pip, que é o
gerenciador de pacotes Python. Vá até a [página oficial do python](https://python.org) e baixe a versão mais nova do instalador
`.msi`. Importante: o site oferece as versões 2 e 3 do Python. Você deve instalar a versão mais recente do Python 2 (na data de
edição desse arquivo essa versão mais nova é a 2.7.13). Quando executar o instalador, você poderá instalar o Python usando o
método "Next, Next, Install" padrão do windows.

É importante, durante a instalação, marcar que você deseja incluir a pasta no Python na sua variável de ambiente PATH. Caso contrário,
você precisar fazer isso manualmente após a instalação.

No Windows, não é possível instalar as bibliotecas `NumPy` e `SciPy` diretamente pelo pip. Você precisa baixar um arquivo
Wheel (com extensão `.whl`). Esses arquivos podem ser encontrados na página da Universidade da Califórnia, Irvine. Para o
Numpy, você pode seguir esse [link](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy). Para o Scipy, clique
[aqui](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy). Esteja atento para baixar as versões dos arquivos para versão 2 do Python
e para a plataforma em que você instalou o Python, x86 ou x64.

Uma vez que você tenha baixado os wheels para o NumPy e o SciPy, você pode abrir um prompt de commando do Windows e mudar para o
diretório do projeto. Uma vez lá, você pode rodar os comandos:

```sh
> pip install <CAMINHO_PARA_NUMPY.WHL>
> pip install <CAMINHO_PARA_SCIPY.WHL>
> pip install -r requirements.txt
```

Lembre-se de substituir os caminhos acima pelos caminhos para os arquivos baixados do site da Universidade da Califórnia, Irvine.
Esses comandso irão instalar as dependências do projeto na sua distribuição Python. Uma vez que as dependências foram instaladas,
você pode instalar o próprio projeto:

```sh
> pip install -e .
```

Isso fará com que o projeto seja instalado em modo `symlink` o que significa que quaisquer modificações feitas no código-fonte serão
refletidas na solução instalada.

## Executando os testes

Para a questão 1, existem testes unitários que comprovam a corretude do código. Você pode checar esses testes executando o seguinte
commando:

```sh
> python setup.py test
```

Se todos os testes passarem, você instalou todas as dependências e o próprio projeto com sucesso. Caso contrário, certifique-se de
que possui uma cópia do projeto atualizada com o ramo `master` desse repositório e que as dependências foram instaladas sem erro.
Caso os testes continuem falhando, por favor, abra uma issue no Github ou entre em contato com `vat@cin.ufpe.br`.

## Executando os experimentos

(Atenção: a depender da potência da sua máquina e do tamanho da base de dados, os experimentos podem demorar várias horas para
concluir a execução.)

### Experimento 1 - Algoritmo de Agrupamento

O primeiro experimento executa o algoritmo de agrupamento na base de dados passada a ele. Ele então, compara a partição obtida com a
partição formada pelas classes originalmente atribuídas a cada objeto. Para rodar esse experimento, execute a seguinte ferramenta, que
você instalou no seu sistema juntamente com o projeto:

```sh
$ amproj_cmd <CAMINHO_PARA_ARQUIVO_DE_DADOS>
```

Estatísticas de execução serão exibidas na tela para ajudá-lo a encontrar gargalos. Se a sua base possuir uma amostra muito grande, o
algoritmo poderá levar muito tempo. Uma vez que o algoritmo tenha sido executado 100 vezes (isso porque ele depende muito da solução
inicial, que é aleatória) a ferramenta irá imprimir na tela:

 * Os grupos que formam a partição _hard_ encontrada;
 * Os vetores de relevância das views.
 * A lista de representantes (ou protótipos) de cada grupo;
 * O valor do índice de rand;
 * O valor do índice de rand corrigido.

Você pode também utilizar a implementação do algoritmo como uma biblioteca Python comum, por exemplo:

```python
>>> from amproj.distance import FuzzyKMedoids
```

As outras funções e classes estão disponíveis de forma semelhante. Você pode observar os testes unitários para ter uma ideia de como
utilizá-las.

### Experimento 2 - Algoritmos Supervisionados

O segundo experimento treina cinco classificadores na base de Segmentação de Imagens disponível no repositório UCI:
 * KNN na View _shape_;
 * KNN na View RGB;
 * Classificador Bayes supondo que as PDFs seguem uma distribuição normal multivariada e utilizando a estimativa da máxima
 verossimilhança para estimar os paramêtros da distribuição. O primeiro Bayes é treinado na view _shape_;
 * O mesmo classificador Bayes acima, porém treinado na view RGB;
 * A combinação dos quatro classificadores acima utilizando a regra do voto majoritário.

Para cada um deles, o experimento retorna:
 * Estimativas pontuais para a média da taxa de erro e para o desvio padrão da taxa de erro;
 * Estimativas intervalares para a média da taxa de erro;
 * Os resultados do teste de Friedman;
 * Os resultados do pós-teste de Nemenyi.

Para rodar o segundo experimento, execute o seguinte commando (supondo que você está na raiz do projeto):

```sh
$ python amproj/ExperimentoQ2.py
```

Mais uma vez, atenção para a quantidade de tempo que o experimento pode tomar.
