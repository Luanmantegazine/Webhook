module.exports = {
  content: [
    './app/interface/index.html',
    './app/interface/src/**/*.{js,jsx,ts,tsx}',
  ],
  theme: { extend: {} },
  plugins: [
  require('@tailwindcss/line-clamp'),
  require('tailwind-scrollbar')({ nocompatible: true })
]
};