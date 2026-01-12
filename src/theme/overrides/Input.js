// ----------------------------------------------------------------------

export default function Input() {
  return {
    MuiInputBase: {
      styleOverrides: {
        root: {
          '& input:-webkit-autofill': {
            transitionDelay: '0s',
            transitionProperty: 'background-color, color',
            WebkitTextFillColor: '#212B36',  // 直接使用颜色值
            '&:active, &:focus, &:hover': {
              WebkitTextFillColor: '#212B36',  // 直接使用颜色值
            },
          },
        },
        input: {
          fontSize: '1rem',  // 使用默认字体大小
          '&::placeholder': {
            opacity: 1,
            color: '#637381',  // 直接使用颜色值
          },
        },
      },
    },
    MuiInput: {
      styleOverrides: {
        underline: {
          '&:before': {
            borderBottomColor: 'rgba(145, 158, 171, 0.32)',  // 直接使用颜色值
          },
        },
      },
    },
    MuiFilledInput: {
      styleOverrides: {
        root: {
          backgroundColor: 'rgba(145, 158, 171, 0.08)',  // 直接使用颜色值
          '&:hover': {
            backgroundColor: 'rgba(145, 158, 171, 0.12)',  // 直接使用颜色值
          },
          '&.Mui-focused': {
            backgroundColor: 'rgba(145, 158, 171, 0.16)',  // 直接使用颜色值
          },
          '&.Mui-disabled': {
            backgroundColor: 'rgba(145, 158, 171, 0.08)',  // 直接使用颜色值
          },
        },
        underline: {
          '&:before': {
            display: 'none',
          },
        },
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-notchedOutline': {
            borderColor: 'rgba(145, 158, 171, 0.24)',  // 直接使用颜色值
          },
          '&.Mui-disabled': {
            '& .MuiOutlinedInput-notchedOutline': {
              borderColor: 'rgba(145, 158, 171, 0.08)',  // 直接使用颜色值
            },
          },
        },
      },
    },
  };
}