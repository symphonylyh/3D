Date: 03/2021

## Install Ubuntu on Jetson

Follow [link](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit) to flash SD card and setup Jetson nano. Recommended SD card is SanDisk Extreme UHS-I U3 based on [link](https://www.accessoriestested.com/best-microsd-card-for-nvidia-jetson-nano/). UHS-II is supported but downgraded to UHS-I.

Was trying to test Gocator 2375 laser profiler and Hokuyo LiDAR devices on Jetson Nano. But later I found these devices are not what we need. Gocator only scans a one-side surface, and Hokuyo single-beam LiDAR only provides a 2D slice of the space.

## Install JAI for line scan camera

* Follow [link](https://www.jai.com/support-software/jetson-ubuntu) to install eBUS JAI. Error encountered: when running the eBUSPlayerJAI script, it has error "error while loading shared libraries: libyaml-cpp.so.0.5: cannot open shared object file: No such file or directory". libyaml-cpp is not installed. The solution is to manually install `libyaml-cpp-dev`. Before that,

```bash
sudo apt-get update
sudo apt-cache madison libyaml-cpp-dev # check available libyaml version, at this time it shows 0.5.2-4ubuntu1
sudo apt-get install libyaml-cpp-dev
```

* Now eBUS for JAI can run normally. Run bash `/opt/jai/ebus_sdk/linux-aarch64-arm/bin/eBUSPlayerJAI` to start the GUI. Put this line in a shell script and `chmod +x` to make it a clickable shortcut. Before that go to Folder System --> Files --> Preferences --> Behavior --> Executable Text Files --> Ask what to do.

## Install ROS for Hokuyo LIDAR

[JetsonHacks](https://www.jetsonhacks.com/) is a good resource for Nano related topics. [F1Tenth](https://f1tenth.org/build.html) is another good resource having steps on setting up ROS. This [blog](https://www.jetsonhacks.com/2019/10/23/install-ros-on-jetson-nano/) provides convenient scripts for installing ROS Melodic on Nano. This [blog](https://www.jetsonhacks.com/2018/02/21/racecar-j-hokuyo-ust-10lx-configuration/) introduces how to use Hokuyo with ROS on Nano. 

## Hokuyo UST-10LX device

Manual and visualization tools can be downloaded on the official [website](https://www.hokuyo-aut.jp/) after login. In addition, more recent version of URG Benri tool can be found [here](http://urgbenri.sourceforge.net/) and [here](https://sourceforge.net/projects/urgbenri/). A [video tutorial](https://www.youtube.com/watch?v=B_es5mJOjmo) for playing with URG Benri.

