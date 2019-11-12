# Reconhecimento Facial
pt/BR | <a href="https://github.com/leoGCoelho/Face-Recognition/blob/master/README.md">en/US</a>

### Resumo
Um sistema de reconhecimento facial usando aprendizado supervisionado (OpenCV + NumPy), analizando conjuntos de imagens e o que foi gravado na webcam.
### Como usar
  - Clonar o Repositório;
  - Incluir (se necessário) mais imagens para treinamento na pasta **images**, marcando a pasta;
  - Na primeira execução, digite o comando `python3 setup.py` para instalar/atualizar todas dependências, treinar o sistema e abrir a camera.
  - Com o sistema treinado, apenas digite `python3 faces.py` no terminal para iniciar a camera;
  - Para treinar o sistema:
    - Use `python3 faces_train.py` para treinar com todas pastas de elementos;
    - Ou use `python3 faces_train_specific.py "NameFolder1 NameFolder2 ... NameFolderX"` para treinar com pastas específicas de elementos;

