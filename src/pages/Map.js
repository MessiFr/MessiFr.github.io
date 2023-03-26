import { 
  Container,
  Button, 
  Grid, 
  Card, 
  CardHeader, 
  Box, 
  CardMedia, 
  CardActions, 
  CardContent,
  Typography
} from "@mui/material";

import { Link } from 'react-router-dom'
import Page from '../components/Page';

export default function Map() {

  return (
    <Page title="Map | COMP90024-Group 21">
      <Container maxWidth="xl">

        <Grid container spacing={2}>
          
          <Grid item xs={12} md={6} lg={6}>
            <Card>
              <CardHeader title="Melbourne Tweets Scatter"  />
              <Box sx={{ p: 3, pb: 1 }} dir="ltr">
                {/* <ReactApexChart type="line" series={} options={} height={364} /> */}
                <CardMedia 
                  component="img" 
                  height="500"
                  image={ require("../_mocks_/scenario1.png") }
                  />
                
                <CardContent> 
                   <Typography variant="body2" color="text.secondary"> 
                      Sentiment Index of twittes in Great Melbourne.
                     </Typography> 
                </CardContent>  
                
                <CardActions style={{ justifyContent: 'space-between' }}>
                  <Button>data analysis</Button>
                  <Button component={Link} to={"/scenario1"}>see the whole map</Button>
                </CardActions>
              </Box>
              </Card>
          </Grid>

          <Grid item xs={14} md={6} lg={6}>
            <Card>
              <CardHeader title="Melbourne Tweets Suburbs Summary"  />
              <Box sx={{ p: 3, pb: 1 }} dir="ltr">
                {/* <ReactApexChart type="line" series={} options={} height={364} /> */}
                <CardMedia 
                  component="img" 
                  height="500"
                  image={ require("../_mocks_/scenario2.png") }
                  />

                <CardContent> 
                   <Typography variant="body2" color="text.secondary"> 
                      The summary for 20 suburbs in Melbourne. Including the indicator of counts, sum of tweets importance and sum of tweets sentiments.
                     </Typography> 
                </CardContent>  

                <CardActions style={{ justifyContent: 'space-between' }}>
                  <Button>data analysis</Button>
                  <Button component={Link} to={"/scenario2"}>see the whole map</Button>
                </CardActions>
                
              </Box>
              </Card>
          </Grid>
          
        </Grid>
        
      </Container>
    </Page>
  );
}
