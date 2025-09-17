const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Add support for .tflite files
config.resolver.assetExts.push(
  // Binary files
  'tflite',
  'bin',
  'txt',
  'jpg',
  'jpeg',
  'png',
  'gif',
  'webp',
  'bmp',
  'tiff',
  'ico',
  'svg'
);

// Remove .tflite from sourceExts to treat it as an asset
config.resolver.sourceExts = config.resolver.sourceExts.filter(ext => ext !== 'tflite');

module.exports = config;