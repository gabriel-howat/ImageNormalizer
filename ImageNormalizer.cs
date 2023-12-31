﻿using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;


namespace ImageNormalizerLibrary
{
    public class ImageNormalizer
    {
        public string? FilePath { get; set; }
        public string? FileName { get; set; }
        public string? OutputFilePath { get; set; }
        public string? OutputFileName { get; set; }
        public string? ImageBase64 { get; set; }
        public Stream? ImageStream { get; set; }

        private Mat originalImage = new();

        private Mat outputImage = new();

        public ImageNormalizer(string base64, string newFileName, string newFilePath)
        {
            ImageBase64 = base64;
            OutputFileName = newFileName;
            OutputFilePath = newFilePath;
        }

        public ImageNormalizer(string base64)
        {
            ImageBase64 = base64;
        }

        public ImageNormalizer(Stream imageStream)
        {
            ImageStream = imageStream;
        }


        public ImageNormalizer(string oldFilePath, string oldFileName, string newFileName, string newFilePath)
        {
            FilePath = oldFilePath;
            FileName = oldFileName;
            OutputFileName = newFileName;
            OutputFilePath = newFilePath;
        }

        private async Task<Mat> PreProcessImageStream(int newHeight)
        {

            if (ImageStream is null)
                throw new NullReferenceException(nameof(ImageStream));

            using var memoryStream = new MemoryStream();
            await ImageStream.CopyToAsync(memoryStream);
            Mat originalMat = new Mat();

            await Task.Run(() => CvInvoke.Imdecode(memoryStream.ToArray(), ImreadModes.Color, originalMat));

            originalMat = ResizeMatByHeight(originalMat, newHeight);

            Mat grayscaleImage = new Mat();
            CvInvoke.CvtColor(originalMat, grayscaleImage, ColorConversion.Bgr2Gray);

            Mat binaryImage = new Mat();
            CvInvoke.Threshold(grayscaleImage, binaryImage, 200, 255, ThresholdType.Binary);
            grayscaleImage.Dispose();

            return binaryImage;
        }

        private async Task<Mat> PreProcessImagePath()
        {
            using (originalImage = await Task.Run(() => CvInvoke.Imread(FilePath + "\\" + FileName)))
            {
                Mat grayscaleImage = new Mat();
                CvInvoke.CvtColor(originalImage, grayscaleImage, ColorConversion.Bgr2Gray);

                Mat binaryImage = new Mat();
                CvInvoke.Threshold(grayscaleImage, binaryImage, 255, 255, ThresholdType.Binary);

                grayscaleImage.Dispose();

                return binaryImage;
            }
        }

        private async Task<Mat> PreProcessImageBase64(int newHeight)
        {
            if (ImageBase64 is null)
                return new();

            byte[] imageBytes = Convert.FromBase64String(ImageBase64);

            // Load the image using a MemoryStream

            // Convert the Bitmap to an Emgu.CV image (Image<Bgr, byte>)
            await Task.Run(() => CvInvoke.Imdecode(imageBytes, ImreadModes.Color, originalImage));

            originalImage = ResizeMatByHeight(originalImage, newHeight);

            Mat grayscaleImage = new Mat();
            CvInvoke.CvtColor(originalImage, grayscaleImage, ColorConversion.Bgr2Gray);

            Mat binaryImage = new Mat();
            CvInvoke.Threshold(grayscaleImage, binaryImage, 200, 255, ThresholdType.Binary);
            //outputBitmap = BitmapExtension.ToBitmap(binaryImage);
            grayscaleImage.Dispose();

            return binaryImage;
        }

        private Mat ResizeMatByHeight(Mat originalMat, int desiredHeight)
        {
            int originalWidth = originalMat.Width;
            int originalHeight = originalMat.Height;
            double aspectRatio = (double)originalWidth / originalHeight;

            int calculatedWidth = (int)(desiredHeight * aspectRatio);

            // Resize the Mat using the calculated width and desired height while preserving the aspect ratio.
            Mat resizedMat = new Mat();
            CvInvoke.Resize(originalMat, resizedMat, new Size(calculatedWidth, desiredHeight), 0, 0, Inter.Linear);

            return resizedMat;
        }

        private async Task<Rectangle> GetBoundingRect(Mat image)
        {
            int top = -1;
            int bottom = -1;
            int left = -1;
            int right = -1;

            // Find the bounding box using parallel loops
            await Task.Run(() =>
            {
                Parallel.For(0, image.Rows, y =>
                {
                    for (int x = 0; x < image.Cols; x++)
                    {
                        if (image.GetRawData(y, x)[0] == 0) // Black pixel found
                        {
                            if (top == -1 || y < top)
                                top = y;
                            if (bottom == -1 || y > bottom)
                                bottom = y;
                            if (left == -1 || x < left)
                                left = x;
                            if (right == -1 || x > right)
                                right = x;
                        }
                    }
                });
            });

            // Get the region of interest (ROI) based on the bounding box
            var boundingBox = new Rectangle(left, top, right - left, bottom - top);

            boundingBox.X = Math.Max(boundingBox.X, 0);
            boundingBox.Y = Math.Max(boundingBox.Y, 0);
            boundingBox.Width = Math.Min(boundingBox.Width, originalImage.Width - boundingBox.X);
            boundingBox.Height = Math.Min(boundingBox.Height, originalImage.Height - boundingBox.Y);

            return boundingBox;
        }

        public Mat GetMatImage()
        {
            return outputImage;
        }

        public async Task<string> GetBase64Image()
        {
            if (outputImage == null)
                throw new InvalidOperationException("No image data available.");
            // Save the image in JPEG format to a MemoryStream
            using (MemoryStream ms = new MemoryStream())
            {
                // Convert the Mat image to a byte array
                byte[] imageBytes = await Task.Run(() => CvInvoke.Imencode(".jpg", outputImage).ToArray());

                // Save the byte array to the MemoryStream
                ms.Write(imageBytes, 0, imageBytes.Length);
                var array = ms.ToArray();

                // Convert the MemoryStream to a base64 string
                return Convert.ToBase64String(array);
            }
        }

        public async Task<Stream> GetImageStreamAsync()
        {
            if (outputImage == null)
                throw new InvalidOperationException("No image data available.");

            using (var memoryStream = new MemoryStream())
            {
                byte[] imageBytes = await Task.Run(() => CvInvoke.Imencode(".jpg", outputImage).ToArray());

                await memoryStream.WriteAsync(imageBytes, 0, imageBytes.Length);
                memoryStream.Seek(0, SeekOrigin.Begin);

                return memoryStream;
            }
        }

        public async Task Normalize(int targetHeight, int finishSize)
        {
            Mat binaryImage;


            if (ImageBase64 != null)
                binaryImage = await PreProcessImageBase64(finishSize);
            else if (!string.IsNullOrEmpty(FilePath) && !string.IsNullOrEmpty(FileName))
                binaryImage = await PreProcessImagePath();
            else if (ImageStream != null)
                binaryImage = await PreProcessImageStream(finishSize);
            else
                throw new InvalidOperationException("No valid input data provided.");


            Rectangle boundingBox = await GetBoundingRect(binaryImage);


            // Get the region of interest (ROI) based on the adjusted bounding box
            using (Mat roiImage = new Mat(originalImage, boundingBox))
            {
                double originalAspectRatio = (double)boundingBox.Width / boundingBox.Height;

                // Set the desired height for the output image
                int targetWidth = (int)(targetHeight * originalAspectRatio);

                // Resize the roiImage to the desired target height while maintaining its aspect ratio
                using (Mat resizedImage = new Mat())
                {
                    CvInvoke.Resize(roiImage, resizedImage, new Size(targetWidth, targetHeight));


                    // Calculate the padding required to center the resized roiImage on the white background
                    int paddingX = (finishSize - resizedImage.Width) / 2;
                    if (paddingX <= 0) paddingX = 0;
                    int paddingY = (finishSize - resizedImage.Height) / 2;
                    if (paddingY <= 0) paddingY = 0;

                    // Create a new white background
                    Mat whiteBackground = new Mat(finishSize, finishSize, originalImage.Depth, originalImage.NumberOfChannels);
                    whiteBackground.SetTo(new MCvScalar(255, 255, 255)); // Set all pixels to white

                    // Paste the resized roiImage onto the white background
                    CvInvoke.CopyMakeBorder(resizedImage, whiteBackground, paddingY, paddingY, paddingX, paddingX, BorderType.Constant, new MCvScalar(255, 255, 255));

                    // Save the final image to the specified output path
                    outputImage = whiteBackground;
                    
                    //outputBitmap = BitmapExtension.ToBitmap(outputImage);
                }
            }
        }
    }
}