import React, {useEffect, useState} from 'react';
import ReactECharts from 'echarts-for-react';
import { styled } from '@mui/material/styles';
import { Card, CardHeader, Select, MenuItem, InputLabel, FormControl } from '@mui/material';
// import colors from '../../../utils/colorSeries';
import SERVER from './config'; 

const CHART_HEIGHT = 450;
const LEGEND_HEIGHT = 72;

const ITEM_HEIGHT = 48;
const ITEM_PADDING_TOP = 8;

const ChartWrapperStyle = styled('div')(({ theme }) => ({
  height: CHART_HEIGHT+50,
  marginTop: theme.spacing(2),
  marginBottom: theme.spacing(5),
  '& .apexcharts-canvas svg': {
    height: CHART_HEIGHT
  },
  '& .apexcharts-canvas svg,.apexcharts-canvas foreignObject': {
    overflow: 'visible'
  },
  '& .apexcharts-legend': {
    height: LEGEND_HEIGHT,
    alignContent: 'center',
    position: 'relative !important',
    borderTop: `solid 1px ${theme.palette.divider}`,
    top: `calc(${CHART_HEIGHT - LEGEND_HEIGHT}px) !important`
  }
}));

const MenuProps = {
    PaperProps: {
      style: {
        maxHeight: ITEM_HEIGHT * 4.5 + ITEM_PADDING_TOP,
        width: 250,
      },
    },
  };

export default function LiveTweets() {

    const [ suburbList, setSuburbList ] = useState([]);

    const [ options, setOptions ] = useState({});
    const [ suburbs, setSuburbs ] = useState([]);

    useEffect(() => {
        fetch(`${SERVER}/api/fields/suburb`, {
          method: 'GET'
        }).then(
          response => response.json()
        ).then(
          data => {
            const FIELDS = [];
            data.data.map((v) => {
              FIELDS.push(v);
              return null;
            });
            setSuburbList(FIELDS);

            if (FIELDS.length > 1){
              setSuburbs([FIELDS[0], FIELDS[FIELDS.length-1]]);
            } else{
              setSuburbs([FIELDS[0]]);
            }
          }
        )
      }, []);

    useEffect(() => {
        let link = `${SERVER}/api/overview/line`;
        if (suburbs.length > 0) {
            link = link + '?';

            for (let i in suburbs) {
                
                let tmp = `sub${i}=${suburbs[i]}`;
                if (!(Number(i) === suburbs.length-1)) {
                    tmp = tmp + '&';
                };
                link = link + tmp;
            };
        }

        fetch(link, {
            method: 'GET'
        }).then(
            response => response.json()
        ).then(
        data => {
            const val = [];
            const show_data = data.data.data;
            Object.keys(show_data).map(v => {
                
                let tmp = {
                        name: v,
                        type: 'line',
                        // stack: 'Total',
                        data: show_data[v],
                    };
                val.push(tmp);
                return null;
            });

            const option = {

                tooltip: {
                    trigger: 'axis'
                  },
                
                legend: {
                    data: suburbs,
                },

                dataZoom: [
                  {
                    show: true,
                    realtime: true,
                    // start: 85,
                    // end: -1,
                  }
                ],
        
                xAxis: {
                //   type: 'time',
                  data: data.data.date_list,
                },
                yAxis: {
                  type: 'value',
                },
                series: val,
            };
        
            setOptions(option);
          }
        )
    }, [suburbs]);
  
  const handleSelect = (event) => {
    // let tmp = suburbs;
    // tmp.push(v);
    const {
        target: { value }
    } = event

    setSuburbs(
        typeof value === 'string' ? value.split(',') : value,
    );
    // setSuburbs(v);
  };

  return (
    <Card>
      <CardHeader title="Sentiment Index (Recently) " />
        <ChartWrapperStyle dir="ltr">
        <ReactECharts option={options} notMerge={true} style={{height: "100%"}}/>
        </ChartWrapperStyle>

        <FormControl
          sx={{ m:1, minWidth: 300, marginLeft: "7%", marginTop: "-1%"}}
        >
          <InputLabel id="aspect-select">suburbs</InputLabel>
          <Select
              label="suburbs"
              value={suburbs}
              onChange={e=>{handleSelect(e)}}
              multiple
              style={{ width:300, height:40, marginLeft:"1%", marginBottom: 20 }}
              MenuProps={MenuProps}
          >
            { suburbList.map(item => (
              <MenuItem key={item} value={item}>{item}</MenuItem>
            ))}
          </Select>
        </FormControl>

    </Card>
  )
}
