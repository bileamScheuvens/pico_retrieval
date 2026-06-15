{
  description = "Devenv.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };
  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        system = system;
        config.allowUnfree = true;
        # config.allowBroken = true;
        config.cudaSupport = true;
      };
      python = pkgs.python3.override {
        packageOverrides = new: old: {
          umap-learn = old.umap-learn.overridePythonAttrs (oldAttrs: {
            doCheck = false;
          });
          biopython = old.biopython.overridePythonAttrs (oldAttrs: {
            doCheck = false;
          });
          onnxscript = old.onnxscript.overridePythonAttrs (oldAttrs: {
            doCheck = false;
          });
        };
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell rec {
        buildInputs = with pkgs; [
          (python.withPackages (
            ps: with ps; [
              joblib
              hydra-core
              omegaconf
              onnxscript
              onnxruntime
              biopython
              ipdb
              plotly
              kaleido
              lightning
              matplotlib
              networkx
              nltk
              pandas
              peft
              pytest
              sentence-transformers
              seqeval
              torch-geometric
              transformers
              umap-learn
              wandb
              pudb
              pip
              pip-tools

              (buildPythonPackage rec {
                pname = "trackio";
                version = "0.26.0";
                pyproject = true;

                src = fetchFromGitHub {
                  owner = "gradio-app";
                  repo = "trackio";
                  tag = "trackio@${version}";
                  hash = "sha256-d2eWN+7lAlcnQwc5RVZvFlYv2CDhEPXZ329al5Rg44g=";
                };

                npmDeps = fetchNpmDeps {
                  src = "${src}/trackio/frontend";
                  hash = "sha256-q1XMYwmQOULuReHnMdeRT4xzd4WOVsll6xdzv2UMgI8=";
                };
                env.SKIP_FRONTEND_BUILD = "1";
                nativeBuildInputs = [
                  nodejs
                  npmHooks.npmConfigHook
                  npmHooks.npmBuildHook
                ];
                npmRoot = "trackio/frontend";

                build-system = [
                  hatchling
                ];

                dependencies = [
                  gradio-client
                  huggingface-hub
                  numpy
                  orjson
                  pillow
                  python-multipart
                  starlette
                  tomli
                  uvicorn
                ];

                optional-dependencies = {
                  apple-gpu = [
                    psutil
                  ];
                  dev = [
                    playwright
                    pytest
                    pytest-playwright
                    ruff
                  ];
                  gpu = [
                    nvidia-ml-py
                    psutil
                  ];
                  mcp = [
                    mcp
                  ];
                  spaces = [
                    pyarrow
                  ];
                };

              })
              (buildPythonPackage rec {
                pname = "NERDA";
                version = "1.0.0";
                format = "pyproject";

                src = fetchFromGitHub {
                  owner = "bileamScheuvens";
                  repo = "NERDA";
                  rev = "13a398c821feee614c1581050c8872486d5a0be4";
                  hash = "sha256-n/xPE26AhA4eFPiCiRUShRvQEOutQJ+MtONUxKA6ngs=";
                };

                build-system = [
                ];

                pythonRemoveDeps = [ "sklearn" ];
                dependencies = [
                  torch
                  transformers
                  scikit-learn
                  nltk
                  pandas
                  progressbar
                  (buildPythonPackage rec {
                    pname = "pyconll";
                    version = "4.0.0";
                    format = "pyproject";
                    src = fetchFromGitHub {
                      owner = "pyconll";
                      repo = "pyconll";
                      tag = "4.0.0";
                      hash = "sha256-SwV4CLwqGJA259nmmIGaAgSfQShzgCFJgt1MmFZTez4=";
                    };
                    build-system = [
                      setuptools
                    ];

                    postPatch = ''
                      substituteInPlace pyproject.toml \
                        --replace '", "util"' '"'
                    '';
                  })
                ];
                postPatch = ''
                  substituteInPlace setup.py \
                    --replace 'setup_requires=["pytest-runner"],' "" \
                    --replace "setup_requires=['pytest-runner']," ""
                '';
              })

            ]
          ))
          cudatoolkit
        ];

        shellHook = ''
          unset SOURCE_DATE_EPOCH
          export CUDA_PATH=${pkgs.cudatoolkit}
          export LD_LIBRARY_PATH=${
            pkgs.lib.makeLibraryPath (
              [
                "/run/opengl-driver" # Needed to find libGL.so
              ]
              ++ buildInputs
            )
          }:$LD_LIBRARY_PATH

          # Set LIBRARY_PATH to help the linker find the CUDA static libraries
          export LIBRARY_PATH=${
            pkgs.lib.makeLibraryPath [
              pkgs.cudatoolkit
            ]
          }:$LIBRARY_PATH
        '';
      };
    };
}
