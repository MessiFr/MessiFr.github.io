import { faker } from '@faker-js/faker';
import PropTypes from 'prop-types';
// material
import { Box, Grid, Card, Paper, Typography, CardContent, Button } from '@mui/material';

// component
import Iconify from '../../../components/Iconify';
import { Icon } from '@iconify/react';
// import { SocialIcon } from 'react-social-icons';


// ----------------------------------------------------------------------

const SOCIALS = [
  {
    name: 'Github',
    value: faker.datatype.number(),
    icon: <Iconify icon="eva:github-outline" width={50} height={50} />
  },
  {
    name: 'Instagram',
    value: faker.datatype.number(),
    icon: <Icon icon="skill-icons:instagram" width={50} height={50}/>
  },
  {
    name: 'LinkedIn',
    value: faker.datatype.number(),
    icon: <Icon icon="icon-park:instagram-one" width={50} height={50} />
  },
  {
    name: 'WeChat',
    value: faker.datatype.number(),
    icon: <Icon icon="ph:wechat-logo-fill" color="#09b83e" width={50} height={50}/>
  },
];

// ----------------------------------------------------------------------

SiteItem.propTypes = {
  site: PropTypes.object
};

function SiteItem({ site }) {
  const { icon, value, name } = site;
  const hasLink = (name!=='');

  console.log(value);
  
  const handle = () =>{
    const w=window.open('about:blank');
    if (name === 'Github') {
      w.location.href="https://github.com/MessiFr";
    } else if (name === 'Instagram'){
      w.location.href="https://www.instagram.com/yuhengfan/";
    } else if (name === 'LinkedIn'){
      w.location.href="https://www.linkedin.com/in/yuheng-fan-b915917b/";
    } else if (name === 'WeChat'){
      w.location.href="https://couchdb.apache.org/";
    } 
  }

  return (
    <Grid item xs={6}>
      <Paper variant="outlined" sx={{ py: 2.5, textAlign: 'center' }}>
        <Box sx={{ mb: 0.5 }}>{icon}</Box>
        <Typography variant="h6">
        {hasLink ? (
          <Button onClick={handle}>
            {name}
          </Button>
        ) : (
          <Button>
            {name}
          </Button>
          )
        }
        </Typography>
      </Paper>
    </Grid>
  );
}

export default function AppTrafficBySite() {
  return (
    <Card>
      {/* <CardHeader title="Project Links" /> */}
      <CardContent>
        <Grid container spacing={2}>
          {SOCIALS.map((site) => (
            <SiteItem key={site.name} site={site} />
          ))}
        </Grid>
      </CardContent>
    </Card>
  );
}
