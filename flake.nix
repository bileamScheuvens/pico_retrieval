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
              biopython
              datasets
              dash
              openai
              faiss
              gradio
              hydra-core
              ipdb
              joblib
              kaleido
              lightning
              matplotlib
              networkx
              nltk
              omegaconf
              onnxruntime
              onnxscript
              pandas
              peft
              pip
              pip-tools
              plotly
              pudb
              pytest
              sentence-transformers
              sentence-transformers.optional-dependencies.image
              seqeval
              torch-geometric
              transformers
              torchvision
              umap-learn
              wandb
              # (buildPythonPackage (finalAttrs: {
              #   pname = "ranx";
              #   version = "v0.3.21";
              #   pyproject = true;
              #   __structuredAttrs = true;
              #
              #   src = fetchFromGitHub {
              #     owner = "AmenRa";
              #     repo = "ranx";
              #     rev = "7363db0c35e92e90d6fa6fe73907b760678f765e";
              #     hash = "sha256-qln64FiYmR/tuS9pzuJZQQGG6VKAdL7nOq/xfgdikmo=";
              #   };
              #
              #   build-system = [
              #     setuptools
              #   ];
              #
              #   dependencies = [
              #     cbor2
              #     fastparquet
              #     (buildPythonPackage (finalAttrs: {
              #       pname = "ir-datasets";
              #       version = "0.5.11";
              #       pyproject = true;
              #       __structuredAttrs = true;
              #       dontCheckRuntimeDeps = true;
              #
              #       src = fetchFromGitHub {
              #         owner = "allenai";
              #         repo = "ir_datasets";
              #         tag = "v${finalAttrs.version}";
              #         hash = "sha256-9RNTs6WiwmFc7LG2LGZuRxUMLHw2RePELCZx/7IF5cQ=";
              #       };
              #
              #       build-system = [
              #         setuptools
              #         wheel
              #       ];
              #
              #       dependencies = [
              #         beautifulsoup4
              #         ijson
              #         inscriptis
              #         lxml
              #         lz4
              #         numpy
              #         pyarrow
              #         pyyaml
              #         requests
              #         tqdm
              #       ];
              #     }))
              #     lz4
              #     numba
              #     numpy
              #     orjson
              #     pandas
              #     rich
              #     scipy
              #     seaborn
              #     tabulate
              #     tqdm
              #
              #   ];
              #
              # }))

              (buildPythonPackage rec {
                pname = "pytrec-eval";
                version = "0.5";
                format = "setuptools";
                # pyproject = true;
                __structuredAttrs = true;

                src = fetchFromGitHub {
                  owner = "cvangysel";
                  repo = "pytrec_eval";
                  tag = version;
                  hash = "sha256-t76D3C5QMJgQMhAg8TGxdtjwaLQhlB8SufAdM3pAZg4=";
                  fetchSubmodules = true;
                };

                build-system = [
                  setuptools
                ];

                dependencies = [
                  numpy
                  scipy
                ];
                NIX_CFLAGS_COMPILE = "-std=gnu17 -Wno-error=implicit-function-declaration -Wno-error=implicit-int -Wno-error=incompatible-pointer-types";
              })

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
          cudaPackages.cuda_nvrtc.lib
          cudaPackages.cuda_nvrtc
          cudaPackages.cudnn
          cudaPackages.libcublas
          cudaPackages.libcufft
          cudaPackages.libcurand
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
          export LD_PRELOAD=${pkgs.cudaPackages.cuda_nvrtc.lib}/lib/libnvrtc.so.12:$LD_PRELOAD
        '';
      };
    };
}
