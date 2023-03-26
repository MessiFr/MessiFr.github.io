import React, {useState} from 'react';

import { 
  Card, 
  Select,
  MenuItem,
  InputLabel,
  FormControl,
  CardHeader, 
  Box, 
  CardMedia,
} from "@mui/material";
import "echarts-wordcloud";

const ITEM_HEIGHT = 48;
const ITEM_PADDING_TOP = 8;

const MenuProps = {
  PaperProps: {
    style: {
      maxHeight: ITEM_HEIGHT * 4.5 + ITEM_PADDING_TOP,
      width: 250,
    },
  },
};

export default function TagCloud() {

  const [ indicator, setIndicator ] = useState("healthy");

  return (
    <Card>
      <CardHeader title="Word Cloud In Suburbs" />
      
        <Box sx={{ p: 3, pb: 1 , margin: 5}} dir="ltr">
          {/* <ReactApexChart type="line" series={} options={} height={364} /> */}
          <CardMedia 
            component="img" 
            height="500"
            image={ require(`../../../_mocks_/${indicator}.png`) }
            />
        </Box>

        <FormControl
          sx={{ m:1, minWidth: 300, marginLeft: "7%", marginTop: "-1%"}}
        >
          <InputLabel id="aspect-select">indicator</InputLabel>
          <Select
              label="indicator"
              value={indicator}
              onChange={e=>{setIndicator(e.target.value)}}
              style={{ width:300, height:40, marginLeft:"1%", marginBottom: 20 }}
              MenuProps={MenuProps}
          >
            
            <MenuItem value={"traffic"}>Traffic</MenuItem>
            <MenuItem value={"healthy"}>Healthy</MenuItem>
            
          </Select>
        </FormControl>

    </Card>
  )
}
