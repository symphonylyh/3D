ffmpeg is a very useful command line application.

### Video to Frame

For multi-view images of a stockpile, we took two sets of data:

* Photos from different viewing angles. Each photo is clear since the operator has manual control of it.
* Videos by moving around the stockpile. Then we need to extract frames from the video to get multi-view images. 

**Important note: on iPhone, Settings -- Camera -- Formats -- Most compatible & Record Video -- 4K at 24 fps.** This is very important! otherwise the raw images will be in HEIC instead of JPG and extracted frames may be of low resolution.

A naive approach extracts frames at certain frame rate, `ffmpeg -i video.MOV -vf fps=2 frames/f%4d.jpg`, but this may lead to some blurred frames. `-vf` means video filter, `fps=2` means extracting two frames per second. Usually we have a video around 30-45 seconds, so 60 to 90 images.

An improved approach, `ffmpeg -i video.MOV -qmin 1 -q:v 1 -vf "fps=2, yadif"  frames/f%4d.jpg`, is to use the best frame quality (ffmpeg comes with a default compression at `-q:v 2`) by setting the [`-q:v` flag](https://stackoverflow.com/a/10234065), and to apply deinterlacing filter [`yadif`](https://superuser.com/a/1274303). Interlaced video (1080i) and progressive video (1080p) are two different techniques of video encoding. It can be observed this improved approach provides better frame quality.

### Video/Images to GIF

See [blog](http://blog.pkh.me/p/21-high-quality-gif-with-ffmpeg.html) 

* From video`ffmpeg -i video.mov -vf scale=480:-1,smartblur=ls=-0.5,crop=iw:ih-2:0:0 -r 5 result.gif`
* From images `ffmpeg -framerate 5 -start_number 1 -i %03d.JPG -vf scale=1080:-1 -frames:v 30 side2.gif`
* `-i` for input, `-r` or `-framerate` for FPS, `-vf` for visual filters, specify resize width:height, smartblur to sharpen images, crop to trim (in pixels) a (iw) by (ih-2) region at position (0,0)
* Glob images `ffmpeg -f image2 -pattern_type glob -i '*_mask.jpg' out.gif`. Start/end images `ffmpeg -framerate 5 -start_number 25 -i img_%4d.jpg -frames:v 50 out.gif`. By default the iterator starts from 0, `-start_number` can set this. `-frames:v XX` can control the range of images to be used.
* To avoid color distortion in GIF, first generate a palette for all input images, `ffmpeg -i image_%02d.png -vf palettegen palette.png`, then use the palette during conversion, `ffmpeg -i image_%02d.png -i palette.png -filter_complex "scale=1080:-1" video.gif`
