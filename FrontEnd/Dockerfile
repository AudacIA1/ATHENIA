# Dockerfile frontend

# Etapa de construcción
FROM node:20 as build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build --prod

# Etapa de producción (servidor ligero)
FROM nginx:alpine

# Copiar los archivos generados por Angular desde dist/frontend-app/browser al contenedor
COPY --from=build /app/dist/frontend-app/browser /usr/share/nginx/html

# Exponer el puerto 80 (por defecto para Nginx)
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
