{
  description = "Devenv.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };
  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        system = system;
        config.allowUnfree = true;
        config.allowBroken = true;
      };
      python = pkgs.python3.override {
        packageOverrides = new: old: {
          torch = pkgs.python3Packages.torchWithCuda;
          huggingface-hub = pkgs.python3Packages.huggingface-hub_0;
          transformers = old.transformers_4.overridePythonAttrs (old: {
            dependencies = map (
              p: if p.pname == "huggingface-hub" then new.huggingface-hub else p
            ) old.dependencies;
          });
          diffusers = old.diffusers.overridePythonAttrs (oldAttrs: {
            dependencies = oldAttrs.dependencies ++ [ old.httpx ];
          });
        };
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell rec {
        buildInputs = with pkgs; [
          (python.withPackages (
            ps: with ps; [
              torch-geometric
              seqeval
              transformers
              peft
              sentence-transformers
              lightning
              joblib
              nltk
              wandb
              pytest
              matplotlib
              pandas
              networkx
              (buildPythonPackage rec {
                pname = "adapters";
                version = "1.3.0";
                format = "setuptools";

                src = fetchFromGitHub {
                  owner = "adapter-hub";
                  repo = "adapters";
                  tag = "v1.3.0";
                  hash = "sha256-1i/0cMhFM5acKwRoAKIOVZaaikJov9C+abDiiW/ZUL0=";
                };

                build-system = [
                  setuptools
                ];
                dependencies = [
                  torch
                  transformers
                ];
              })
              (buildPythonPackage rec {
                pname = "NERDA";
                version = "1.0.0";
                format = "pyproject";

                src = fetchFromGitHub {
                  owner = "ebanalyse";
                  repo = "NERDA";
                  rev = "ae45d7e5368059721d1073384201433ea7a6e820";
                  hash = "sha256-EAjxUmqXcSgfKp1E4zeuKGcewSvJuNIjrUv87O/EKVU=";
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
