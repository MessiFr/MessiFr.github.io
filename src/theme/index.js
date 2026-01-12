import { createTheme, responsiveFontSizes } from '@mui/material/styles';
import { zhCN } from '@mui/material/locale';

import palette from './palette';
import typography from './typography';
import breakpoints from './breakpoints';
import shadows, { customShadows } from './shadows';
import shape from './shape';

// Import overrides
import Button from './overrides/Button';
import Input from './overrides/Input';

// ----------------------------------------------------------------------

const theme = createTheme({
  palette,
  typography,
  breakpoints,
  shape,
  shadows,
  customShadows,  // 添加自定义阴影
  direction: 'ltr',
  mixins: {
    toolbar: {
      minHeight: 56,
      paddingTop: 4,
      paddingBottom: 4
    }
  },
  components: {
    ...Button(),
    ...Input(),
    MuiCssBaseline: {
      styleOverrides: {
        '*': {
          margin: 0,
          padding: 0,
          boxSizing: 'border-box',
        },
        html: {
          width: '100%',
          height: '100%',
          WebkitOverflowScrolling: 'touch',
        },
        body: {
          width: '100%',
          height: '100%',
        },
        '#root': {
          width: '100%',
          height: '100%',
        },
        input: {
          '&[type=number]': {
            MozAppearance: 'textfield',
            '&::-webkit-outer-spin-button': {
              margin: 0,
              WebkitAppearance: 'none',
            },
            '&::-webkit-inner-spin-button': {
              margin: 0,
              WebkitAppearance: 'none',
            },
          },
        },
        img: {
          maxWidth: '100%',
          height: 'auto',
          display: 'inline-block',
        },
      },
    },
  },
}, zhCN);

export default responsiveFontSizes(theme);