# Getting Over It RL

We trained our neural-network policy from scratch with reinforcement learning. It started with random weights and learned hammer control from rewards for route progress. The live run uses 106 inputs covering the climber, hammer, cursor, contacts, motor state, terrain probes, and elapsed time. One shared policy learns across 2,048 parallel attempts starting from the bottom.

This repository contains the original fullscreen policy preview and its Windows launcher. The chart shows live route distance, a running timer to 870 m, and the fastest verified preview time. The timer stops at the first crossing. The lower panel shows actual activations, sampled network weights, and the mouse outputs.

After a win, press **R** in the game window to respawn at the bottom. Press **R** again to start the climb and timer. Falls and stalls still restart automatically.

## Use the preview

Use these files in your configured local Getting Over It RL training checkout. The viewer reads the locally trained checkpoint at `runs/<run>/checkpoints/latest.pkl`. It requires a WSL distribution named `kali-linux` and the Python environment at `~/.venvs/goirl-gpu`. Build the HUD with your local Unity bridge. The bridge must support the uppercase `R` protocol command and expose `preview_restart_pressed()` in its Python client.

From the local training checkout in PowerShell:

```powershell
.\Watch-GPU.cmd --run runs/garden-split
```

To close the preview while training continues:

```powershell
New-Item -ItemType File -Force .\runtime\PREVIEW_STOP
```

The release contains preview source. Run data, checkpoints, and game files stay local.

## License

[MIT](LICENSE) © 2026 Samik Swarup Biswal.
