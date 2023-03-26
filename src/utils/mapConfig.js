var mapConfig1 = {
  "version": "v1",
  "config": {
    "visState": {
      "filters": [],
      "layers": [
        {
          "id": "8svgpy",
          "type": "point",
          "config": {
            "dataId": "melbourne_twitter",
            "label": "sentiment",
            "color": [
              207,
              23,
              80
            ],
            "highlightColor": [
              252,
              242,
              26,
              255
            ],
            "columns": {
              "lat": "lat",
              "lng": "long",
              "altitude": null
            },
            "isVisible": true,
            "visConfig": {
              "radius": 30,
              "fixedRadius": false,
              "opacity": 0.8,
              "outline": false,
              "thickness": 2,
              "strokeColor": null,
              "colorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "strokeColorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "radiusRange": [
                0,
                50
              ],
              "filled": true
            },
            "hidden": false,
            "textLabel": [
              {
                "field": null,
                "color": [
                  255,
                  255,
                  255
                ],
                "size": 18,
                "offset": [
                  0,
                  0
                ],
                "anchor": "start",
                "alignment": "center"
              }
            ]
          },
          "visualChannels": {
            "colorField": {
              "name": "sentiment",
              "type": "real"
            },
            "colorScale": "quantile",
            "strokeColorField": null,
            "strokeColorScale": "quantile",
            "sizeField": null,
            "sizeScale": "linear"
          }
        }
      ],
      "interactionConfig": {
        "tooltip": {
          "fieldsToShow": {
            "melbourne_twitter": [
              {
                "name": "time",
                "format": null
              },
              {
                "name": "long",
                "format": null
              },
              {
                "name": "importance",
                "format": null
              },
              {
                "name": "sentiment",
                "format": null
              }
            ]
          },
          "compareMode": false,
          "compareType": "absolute",
          "enabled": true
        },
        "brush": {
          "size": 0.5,
          "enabled": false
        },
        "geocoder": {
          "enabled": false
        },
        "coordinate": {
          "enabled": false
        }
      },
      "layerBlending": "normal",
      "splitMaps": [],
      "animationConfig": {
        "currentTime": null,
        "speed": 1
      }
    },
    "mapState": {
      "bearing": 0,
      "dragRotate": false,
      "latitude": -37.842942449999995,
      "longitude": 145.0987541331105,
      "pitch": 0,
      "zoom": 9,
      "isSplit": false
    },
    "mapStyle": {
      "styleType": "dark",
      "topLayerGroups": {},
      "visibleLayerGroups": {
        "label": true,
        "road": true,
        "border": false,
        "building": true,
        "water": true,
        "land": true
      },
      "threeDBuildingColor": [
        9.665468314072013,
        17.18305478057247,
        31.1442867897876
      ],
      "mapStyles": {}
    }
  }
}

export default mapConfig1;