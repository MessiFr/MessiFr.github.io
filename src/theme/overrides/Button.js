// ----------------------------------------------------------------------

export default function Button() {
  return {
    MuiButton: {
      styleOverrides: {
        root: {
          '&:hover': {
            boxShadow: 'none',
          },
        },
        sizeLarge: {
          height: 48,
        },
        containedInherit: {
          color: '#212B36',  // 直接使用颜色值而不是theme.palette.grey[800]
          boxShadow: '0 8px 16px 0 rgba(145, 158, 171, 0.24)',
          '&:hover': {
            backgroundColor: '#91A3B0',  // 直接使用颜色值
          },
        },
        containedPrimary: {
          boxShadow: '0 8px 16px 0 rgba(16, 82, 204, 0.24)',  // 使用主题主色
          '&:hover': {
            backgroundColor: '#003cab',  // 直接使用颜色值
          },
        },
        containedSecondary: {
          boxShadow: '0 8px 16px 0 rgba(51, 102, 255, 0.24)',  // 使用主题次色
          '&:hover': {
            backgroundColor: '#1939B7',  // 直接使用颜色值
          },
        },
        outlinedInherit: {
          border: '1px solid rgba(145, 158, 171, 0.2)',  // 直接使用颜色值
          '&:hover': {
            backgroundColor: 'rgba(145, 158, 171, 0.08)',  // 直接使用颜色值
          },
        },
        textInherit: {
          '&:hover': {
            backgroundColor: 'rgba(145, 158, 171, 0.08)',  // 直接使用颜色值
          },
        },
      },
    },
  };
}