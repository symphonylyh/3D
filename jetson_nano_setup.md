* Follow [link](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit) to flash SD card and setup Jetson nano. Recommended SD card is SanDisk Extreme UHS-I U3 based on [link](https://www.accessoriestested.com/best-microsd-card-for-nvidia-jetson-nano/). UHS-II is supported but downgraded to UHS-I.
* Follow [link](https://www.jai.com/support-software/jetson-ubuntu) to install eBUS JAI. Error encountered: when running the eBUSPlayerJAI script, it has error "error while loading shared libraries: libyaml-cpp.so.0.5: cannot open shared object file: No such file or directory". libyaml-cpp is not installed. The solution is to manually install `libyaml-cpp-dev`. Before that,

```bash
sudo apt-get update
sudo apt-cache madison libyaml-cpp-dev # check available libyaml version, at this time it shows 0.5.2-4ubuntu1
sudo apt-get install libyaml-cpp-dev
```

* Now eBUS for JAI can run normally. Run bash `/opt/jai/ebus_sdk/linux-aarch64-arm/bin/eBUSPlayerJAI` to start the GUI. Put this line in a shell script and `chmod +x` to make it a clickable shortcut. Before that go to Folder System --> Files --> Preferences --> Behavior --> Executable Text Files --> Ask what to do.