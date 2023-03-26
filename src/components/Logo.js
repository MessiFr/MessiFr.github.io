import { Link as RouterLink } from 'react-router-dom';
// material
import { Box } from '@mui/material';

// ----------------------------------------------------------------------

export default function Logo() {
  return (
    <RouterLink to="/">
      <Box 
        component="img" 
        src="/static/icons/avatar.jpeg" 
        sx={{
          height: 150,
          width: 150
        }}/>
    </RouterLink>
  );
}
