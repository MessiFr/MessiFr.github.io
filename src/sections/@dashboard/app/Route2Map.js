import { 
  Button,
  Card, 
  CardHeader, 
  Box, 
  CardMedia, 
  CardActions, 
  CardContent,
  Typography
} from "@mui/material";

import { Link } from 'react-router-dom'


export default function Route2Map() {
  return (
    <Card>
      <CardHeader title="Map"  />
      <Box sx={{ p: 3, pb: 1 }} dir="ltr">
        {/* <ReactApexChart type="line" series={} options={} height={364} /> */}
        <CardMedia 
          component="img" 
          height="305"
          image={ require("../../../_mocks_/kepler.png") }
          />

        <CardContent> 
            <Typography variant="body2" color="text.primary"> 
              Check the data visualization in Melbourne map for more details.
              </Typography> 
        </CardContent>  

        <CardActions style={{ justifyContent: 'space-between' }}>
          <Button component={Link} to={"/COMP90024_Group21/map"}>go to map</Button>
        </CardActions>
        
      </Box>
    </Card>
  );
}
