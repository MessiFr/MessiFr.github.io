import React, {useEffect, useState} from 'react';
import ReactECharts from 'echarts-for-react';
import { styled } from '@mui/material/styles';
import { Card, CardHeader, Select, MenuItem, InputLabel, FormControl } from '@mui/material';
import colors from 'src/utils/colorSeries';
import SERVER from './config'; 

const CHART_HEIGHT = 1000;
const LEGEND_HEIGHT = 72;

const ChartWrapperStyle = styled('div')(({ theme }) => ({
  height: CHART_HEIGHT,
  marginTop: theme.spacing(2),
  marginBottom: theme.spacing(-10),
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

export default function BarChart() {
  
//   const [chartData, setChartData ] = useState([]);
//   const [suburbs, setSuburbs] = useState([]);
  const [aspect, setAspect] = useState("traffic");
  const [option, setOption] = useState({});
  const [reload, setReload] = useState(true);

  useEffect(() => {
    fetch(`${SERVER}/api/overview/bar/${aspect}`, {
      method: 'GET',
    }).then(
      response => response.json()
    ).then(
      data => {
        
        const options = {
            tooltip: {
              trigger: 'axis',
              axisPointer: {
                type: 'shadow' // 'shadow' as default; can also be 'line' or 'shadow'
              }
            },
            legend: {},
            grid: {
              left: '3%',
              right: '4%',
              bottom: '3%',
              containLabel: true
            },
            xAxis: {
              type: 'value'
            },
            yAxis: {
              type: 'category',
              data:[],
            },
            series: [],
          };
        
        options.yAxis.data = data.data.fields;
        console.log(data.data.fields);
        
        let color_index = colors.length - 1;
        data.data.bar_data.map((v) => {
          
          for (let i in v) {
              const s = {
                            name: i,
                            type: 'bar',
                            stack: 'total',
                            color: colors[color_index],
                            label: {
                                show: true
                            },
                            emphasis: {
                                focus: 'series'
                            },
                            data: v[i],
                        };
                options.series.push(s);
          };
          color_index = color_index - 1;
          return null;
        });
        
        setOption(options);
        setReload(true);
        
      }
    )
  }, [aspect]);

  useEffect(() => {
      if (reload) {
          setReload(false)
      }
  }, [reload]);

  return (
    <Card>
      <CardHeader title="Tweets Counts Of Suburb In Melbourne Overview" />
        <ChartWrapperStyle dir="ltr">
        <ReactECharts option={option} notMerge={true} style={{height: "90%"}}/>
        </ChartWrapperStyle>
        <FormControl
          sx={{ m:1, minWidth: 120, marginLeft: "40%"}}
        >
          <InputLabel id="aspect-select">aspect</InputLabel>
          <Select
              label="aspect"
              value={aspect}
              onChange={e=>{setAspect(e.target.value)}}
              style={{ width:200, height:40, marginLeft:"1%", marginBottom: 20 }}
          >
            <MenuItem value={"traffic"}>traffic</MenuItem>
            <MenuItem value={"healthy"}>healthy</MenuItem>
            
          </Select>
        </FormControl>
    </Card>
  )
}
