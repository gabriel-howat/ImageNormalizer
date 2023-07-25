using System.Drawing;
using System;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using static System.Net.Mime.MediaTypeNames;

namespace ImageNormalizerLibrary
{
    public class ImageNormalizer
    {
        public string? FilePath { get; set; }
        public string? FileName { get; set; }
        public string? OutputFilePath { get; set; }
        public string? OutputFileName { get; set; }
        public string? ImageBase64 { get; set; }

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

        public ImageNormalizer(string oldFilePath, string oldFileName, string newFileName, string newFilePath)
        {
            FilePath = oldFilePath;
            FileName = oldFileName;
            OutputFileName = newFileName;
            OutputFilePath = newFilePath;
        }

        private async Task<Mat> PreProcessImagePath()
        {
            using (originalImage = await Task.Run(() => CvInvoke.Imread(FilePath + "\\" + FileName)))
            {
                Mat grayscaleImage = new Mat();
                CvInvoke.CvtColor(originalImage, grayscaleImage, ColorConversion.Bgr2Gray);

                Mat binaryImage = new Mat();
                CvInvoke.Threshold(grayscaleImage, binaryImage, 200, 255, ThresholdType.Binary);

                grayscaleImage.Dispose();

                return binaryImage;
            }
        }

        private async Task<Mat> PreProcessImageBase64()
        {
            if (ImageBase64 is null)
                return new();

            byte[] imageBytes = Convert.FromBase64String(ImageBase64);

            // Create a MemoryStream from the byte array
            await Task.Run(() => CvInvoke.Imdecode(imageBytes, ImreadModes.Color, originalImage));

            Mat grayscaleImage = new Mat();
            CvInvoke.CvtColor(originalImage, grayscaleImage, ColorConversion.Bgr2Gray);

            Mat binaryImage = new Mat();
            CvInvoke.Threshold(grayscaleImage, binaryImage, 200, 255, ThresholdType.Binary);

            originalImage.Dispose();
            grayscaleImage.Dispose();

            return binaryImage;
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
                object lockObj = new object();
                Parallel.For(0, image.Rows, y =>
                {
                    for (int x = 0; x < image.Cols; x++)
                    {
                        if (image.GetRawData(y, x)[0] == 0) // Black pixel found
                        {
                            lock (lockObj)
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
                    }
                });
            });

            // Get the region of interest (ROI) based on the bounding box
            var boundingBox =  new Rectangle(left, top, right - left, bottom - top);

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
            // Save the image in JPEG format to a MemoryStream
            using (MemoryStream ms = new MemoryStream())
            {
                // Convert the Mat image to a byte array
                byte[] imageBytes = await Task.Run(() => CvInvoke.Imencode(".jpg", outputImage).ToArray());

                // Save the byte array to the MemoryStream
                ms.Write(imageBytes, 0, imageBytes.Length);

                // Convert the MemoryStream to a base64 string
                string base64Image = Convert.ToBase64String(ms.ToArray());

                return base64Image;
            }
        }

        public async Task Normalize(int targetHeight, int finishSize)
        {
            Mat binaryImage;

            if (ImageBase64 != null)
                binaryImage = await PreProcessImageBase64();
            else
                binaryImage = await PreProcessImagePath();


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
                    int paddingY = (finishSize - resizedImage.Height) / 2;

                    // Create a new white background
                    using (Mat whiteBackground = new Mat(finishSize, finishSize, originalImage.Depth, originalImage.NumberOfChannels))
                    {
                        whiteBackground.SetTo(new MCvScalar(255, 255, 255)); // Set all pixels to white

                        // Paste the resized roiImage onto the white background
                        CvInvoke.CopyMakeBorder(resizedImage, whiteBackground, paddingY, paddingY, paddingX, paddingX, BorderType.Constant, new MCvScalar(255, 255, 255));

                        // Save the final image to the specified output path
                        outputImage = whiteBackground;
                    }
                }
            }
        }
    }
}