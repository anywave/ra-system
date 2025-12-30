/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ra: {
          gold: '#D4AF37',
          blue: '#1E3A5F',
          purple: '#6B21A8',
          glow: '#00D4FF',
        }
      }
    },
  },
  plugins: [],
}
