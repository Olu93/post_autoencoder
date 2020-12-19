# ## Background

This part of the series needs a bit more explaination. So far, there haven't been many attempts to use CNN's for 3D objects. Why? Probably because of the representation that would be required for a vanilla CNN to work without many changes.

### Pixels and Voxels

The logical 3D extenstion of a 2D image would be a 3D grid, because an image is essentially a 2D grid with each cell representing a pixel. In the same manner 3D cells in a 3D grid are called voxels. In both cases, pixels and voxels are somewhat abstract containers that hold values. We typically think pixels strictly contain only color values but they can hold much more information. Opacity is an intuitive example, but we could even save meta-data in there. That was just for the intuition. Voxels are pretty common within neuroscience and obviously game technology.

### Voxels and Convolutions

After this description it might be trivial, but we can extend the intuition about convolutions easily onto the next dimension. While we convolve for with a restricted set of 2D filters over images, we do the same with 3D objects. This time, we only imagine cubes instead of squares. In fact, many of you who have already used CNN's know, that this is already common practice, as we treat colors often as that third dimension. This flexibility ties back to the idea that these cells are merely containers and the dimensions can represent anything.

### So far so gut... But what's the issue with those?

Well, neither am I well acquainted with Game-Technology nor do I have a background in Computer Graphics. However, from what I understand, there are about X reasons:

- First, voxel data is exponentially larger than pixel data, given the added dimension. Meaning, that you now need to deal with more data and all of it. If you just care about simple singular 3D models, most of the volume might be empty.

- Second, depending on your resolution (which directly affects the data volume), things can become quite minecrafty.

- Third, there are alternatives: Point clouds and polygons. Both are vector-based and the ladder even maintains structure.

### Why still use voxels then?

They obviously also have advantages. They maintain structure, they are easy to process in computer graphics, they are great for procedural generation. Some of the algorythms for images like JPEG are easy to extend.

### Links

For more about these matters checkout following materials:

- [https://medium.com/@EightyLevel/how-voxels-became-the-next-big-thing-4eb9665cd13a](An explanation for voxels imminent comeback in Gaming. May be a bit difficult to follow.)
- [https://www.quora.com/What-are-the-pros-and-cons-of-using-voxels-instead-of-polygons](Best explanation that I found on this topic. Also videogame centric.)
