{
  description = "digit classifier";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            uv
            gcc
            pkg-config
          ];
          buildInputs = with pkgs; [
            libX11
            cairo
            libGL
            mesa
            pango
            glib
            fontconfig
            freetype
            libXext
            libXrender
            libICE
            libSM
            libxkbcommon
            dbus
            libxcb-cursor
            qt6.qtwayland
            wayland
            libxkbcommon
          ];
          env.QT_QPA_PLATFORM = "wayland";
          shellHook = ''
            export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${
              pkgs.lib.makeLibraryPath [
                pkgs.glib
                pkgs.fontconfig
                pkgs.freetype
                pkgs.libX11
                pkgs.libXext
                pkgs.libXrender
                pkgs.libICE
                pkgs.libSM
                pkgs.libxkbcommon
                pkgs.dbus
                pkgs.libGL
                pkgs.libxcb-cursor
                pkgs.qt6.qtwayland
                pkgs.wayland
                pkgs.libxkbcommon
                pkgs.pkg-config
                pkgs.ffmpeg
              ]
            }:$LD_LIBRARY_PATH"
            source .venv/bin/activate
          '';
        };
      }
    );
}
