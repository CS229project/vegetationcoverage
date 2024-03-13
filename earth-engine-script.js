//var geometry = /* color: #d6d6d6 */ /* shown: false */ /* displayProperties: [ { "type": "rectangle" } ] */ ee.Geometry.Polygon( [[[-60.1348375471504, -2.9180780426389026], [-60.1348375471504, -3.197830219067357], [-59.83957997879102, -3.197830219067357], [-59.83957997879102, -2.9180780426389026]]], null, false);

var geometry = /* color: #d63000 */ee.Geometry.Polygon(
  [[[-60.11428372084107, -3.0549830736880934],
    [-60.0497390431067, -3.131775308509979],
    [-59.9893142384192, -3.1619421534821206],
    [-59.9508620899817, -3.152343706675114],
    [-59.85885159193482, -3.082409509602168],
    [-59.84374539076295, -3.0618397486009923],
    [-59.84786526380982, -2.9342985158694614],
    [-60.1266433399817, -2.9370414934689872]]]);

var startYear = 2000;
var type = "NDVI";



for (var i = 0; i < 23; i++){

var startDate = startYear + '-07';
var endDate = startYear + '-08';


if (type == "NDVI"){
var dataset = ee.ImageCollection('MODIS/061/MOD13Q1')
                .select('NDVI')
                //.filter(ee.Filter.date('2001-01-01', '2001-05-01')
                .filter(ee.Filter.date(startDate, endDate)
                );
var image = dataset.first().clip(geometry);

// Retrieve the projection information from a band of the original image.
// Call getInfo() on the projection to request a client-side object containing
// the crs and transform information needed for the client-side Export function.
var projection = image.select('NDVI').projection().getInfo();

var vis = {
min: 0,
max: 8000,
/*palette: [
  'ffffff', 'ce7e45', 'df923d', 'f1b555', 'fcd163', '99b718', '74a901',
  '66a000', '529400', '3e8601', '207401', '056201', '004c00', '023b01',
  '012e01', '011d01', '011301'
],*/
palette: [
  'ffffff', 'ce7e45', 'df923d', 'f1b555', 'fcd163', '99b718', '74a901',
  '66a000', '529400', '3e8601', '207401', '056201', '004c00', '023b01',
  '012e01', '011d01', '011301'
],

};

} else{

var dataset = ee.ImageCollection('MODIS/061/MOD11A2')
              .filter(ee.Filter.date('2021-07', '2021-08'));
var image = dataset.select('LST_Day_1km').first().clip(geometry);

var projection = image.select('LST_Day_1km').projection().getInfo();

var vis = {
min: 14000.0,
max: 16000.0,
/*palette: [
  '040274', '040281', '0502a3', '0502b8', '0502ce', '0502e6',
  '0602ff', '235cb1', '307ef3', '269db1', '30c8e2', '32d3ef',
  '3be285', '3ff38f', '86e26f', '3ae237', 'b5e22e', 'd6e21f',
  'fff705', 'ffd611', 'ffb613', 'ff8b13', 'ff6e08', 'ff500d',
  'ff0000', 'de0101', 'c21301', 'a71001', '911003'
],*/
palette: [
  '040274', 'ff6e08', 'ff500d',
  'ff0000'
],
};
}


Map.setCenter(-60.0217, -3.1190, 10);
Map.addLayer(image, vis, type);
//Map.addLayer(ndvi);

Export.image.toDrive({
image: image.visualize(vis),
description: type + '_image_export_crstransform_' + startYear,
folder: 'my-drive',
fileNamePrefix: type + "_landsat_manaus_" + startYear,
crs: projection.crs,
crsTransform: projection.transform,
scale: 100
});

startYear = startYear + 1;

}