import cv2
import numpy as np
import os
import re
import unicodedata

#Config
CELL_SIZE = 48            
UMBRAL_SIMILITUD = 0.70   
MARGEN_CELDA = 0.23       

#Cargar las letras de referencia
def cargar_letras_referencia(carpeta_letras='letras'):
    letras = {}

    if not os.path.isdir(carpeta_letras):
        print(f"[Error] Carpeta '{carpeta_letras}' no encontrada.")
        return letras

    for archivo in os.listdir(carpeta_letras):
        if archivo.lower().endswith('.png'):
            letra = archivo.split('.')[0].upper()
            ruta = os.path.join(carpeta_letras, archivo)

            img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"No se pudo cargar {archivo}")
                continue

            img_proc = preprocesar_celda(img)
            if img_proc is not None:
                letras[letra] = img_proc
            else:
                print(f"No se pudo procesar {archivo}")

    return letras

#Preprocesado
def preprocesar_celda(celda):
    if celda is None or celda.size == 0:
        return None

    #Poner en escala de grises
    if len(celda.shape) == 3:
        celda = cv2.cvtColor(celda, cv2.COLOR_BGR2GRAY)

    #Quitar ruido ligero
    desenfocada = cv2.GaussianBlur(celda, (3, 3), 0)

    #Poner las letras claras sobre fondo oscuro
    _, mask = cv2.threshold(desenfocada,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)

    #Calcular los margenes
    margen = 1
    x = max(0, x - margen)
    y = max(0, y - margen)
    w = min(mask.shape[1] - x, w + 2 * margen)
    h = min(mask.shape[0] - y, h + 2 * margen)

    if w <= 0 or h <= 0:
        return None

    roi = desenfocada[y:y + h, x:x + w]
    if roi.size == 0:
        return None
    roi_inv = 255 - roi

    #Normalizar contraste local
    min_val, max_val, _, _ = cv2.minMaxLoc(roi_inv)
    if max_val <= min_val:
        return None

    roi_norm = (roi_inv - min_val) * (255.0 / (max_val - min_val))
    roi_norm = roi_norm.astype(np.uint8)

    # Ajustamos a CELL_SIZE x CELL_SIZE manteniendo proporción
    h0, w0 = roi_norm.shape
    escala = min(CELL_SIZE / float(w0), CELL_SIZE / float(h0))
    new_w = max(1, int(round(w0 * escala)))
    new_h = max(1, int(round(h0 * escala)))

    letra_resized = cv2.resize(roi_norm,(new_w, new_h),interpolation=cv2.INTER_AREA)

    lienzo = np.zeros((CELL_SIZE, CELL_SIZE), dtype=np.uint8)
    x_off = (CELL_SIZE - new_w) // 2
    y_off = (CELL_SIZE - new_h) // 2
    lienzo[y_off:y_off + new_h, x_off:x_off + new_w] = letra_resized

    lienzo_norm = lienzo.astype(np.float32) / 255.0
    return lienzo_norm

#Similitud y reconocimiento
def similitud_coseno(a, b):
    v1 = a.reshape(-1).astype(np.float32)
    v2 = b.reshape(-1).astype(np.float32)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))

def reconocer_letra(celda, letras_referencia):
    celda_prep = preprocesar_celda(celda)
    if celda_prep is None:
        return '?', 0.0

    mejor_letra = '?'
    mejor_sim = -1.0

    for letra, template in letras_referencia.items():
        if template is None or template.shape != celda_prep.shape:
            continue
        sim = similitud_coseno(celda_prep, template)
        if sim > mejor_sim:
            mejor_sim = sim
            mejor_letra = letra

    if mejor_sim < UMBRAL_SIMILITUD:
        return '?', mejor_sim

    return mejor_letra, mejor_sim

def agrupar_coords(coords, tolerancia=10):
    if not coords:
        return []
    coords = sorted(coords)
    agrupadas = [coords[0]]
    for c in coords[1:]:
        if c - agrupadas[-1] > tolerancia:
            agrupadas.append(c)
    return agrupadas

#Detectar cuadricula
def detectar_y_reconocer(imagen_path, letras_referencia):
    img = cv2.imread(imagen_path)
    if img is None:
        print(f"Error: No se pudo cargar {imagen_path}")
        return

    resultado = img.copy()
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Binarización para detectar solo líneas de la cuadrícula
    binaria = cv2.adaptiveThreshold(gris,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,5)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    lineas_h = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel_h)
    lineas_v = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel_v)

    lh = cv2.HoughLinesP(lineas_h, 1, np.pi / 180,
                         100, minLineLength=80, maxLineGap=10)
    lv = cv2.HoughLinesP(lineas_v, 1, np.pi / 180,
                         100, minLineLength=80, maxLineGap=10)

    y_coords = set()
    x_coords = set()

    if lh is not None:
        for l in lh:
            x1, y1, x2, y2 = l[0]
            y_coords.add(y1)
    if lv is not None:
        for l in lv:
            x1, y1, x2, y2 = l[0]
            x_coords.add(x1)

    y_coords = agrupar_coords(list(y_coords), tolerancia=5)
    x_coords = agrupar_coords(list(x_coords), tolerancia=5)

    celdas = []
    for i in range(len(y_coords) - 1):
        for j in range(len(x_coords) - 1):
            x = x_coords[j]
            y = y_coords[i]
            w = x_coords[j + 1] - x
            h = y_coords[i + 1] - y
            if w > 15 and h > 15:
                celdas.append((x, y, w, h))

    for idx, (x, y, w, h) in enumerate(celdas):
        mx = int(w * MARGEN_CELDA)
        my = int(h * MARGEN_CELDA)

        x_ini = x + mx
        y_ini = y + my
        x_fin = x + w - mx
        y_fin = y + h - my

        if x_fin <= x_ini or y_fin <= y_ini:
            continue

        celda_img = gris[y_ini:y_fin, x_ini:x_fin]
        letra, sim = reconocer_letra(celda_img, letras_referencia)

        cv2.rectangle(resultado, (x, y), (x + w, y + h), (0, 255, 0), 1)
        escala_fuente = min(w, h) / 40.0
        cv2.putText(resultado, letra, (x + 3, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, escala_fuente, (255, 0, 0), 1)

        if idx < 80:
            print(f"Celda {idx+1}: '{letra}' (cos:{sim:.3f})")

    cv2.imshow('Original', img)
    cv2.imshow('Resultado con letras', resultado)

    return resultado

def extraer_matriz_letras(imagen_path, letras_referencia):
    img = cv2.imread(imagen_path)
    if img is None:
        print(f"Error: No se pudo cargar {imagen_path}")
        return None, None
    
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binaria = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 15, 5)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    lineas_h = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel_h)
    lineas_v = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel_v)
    
    lh = cv2.HoughLinesP(lineas_h, 1, np.pi / 180, 100, minLineLength=80, maxLineGap=10)
    lv = cv2.HoughLinesP(lineas_v, 1, np.pi / 180, 100, minLineLength=80, maxLineGap=10)
    
    y_coords = set()
    x_coords = set()
    
    if lh is not None:
        for l in lh:
            x1, y1, x2, y2 = l[0]
            y_coords.add(y1)
    if lv is not None:
        for l in lv:
            x1, y1, x2, y2 = l[0]
            x_coords.add(x1)
    
    y_coords = agrupar_coords(list(y_coords), tolerancia=5)
    x_coords = agrupar_coords(list(x_coords), tolerancia=5)
     
    filas = len(y_coords) - 1
    columnas = len(x_coords) - 1
    
    print(f"Cuadrícula detectada: {filas}x{columnas}")
    
    #Crear matriz de letras 
    matriz = []
    
    for i in range(filas):
        fila_letras = []
        for j in range(columnas):
            x = x_coords[j]
            y = y_coords[i]
            w = x_coords[j + 1] - x
            h = y_coords[i + 1] - y
            
            if w <= 15 or h <= 15:
                fila_letras.append('?')
                continue
            
            mx = int(w * MARGEN_CELDA)
            my = int(h * MARGEN_CELDA)
            
            x_ini = x + mx
            y_ini = y + my
            x_fin = x + w - mx
            y_fin = y + h - my
            
            if x_fin <= x_ini or y_fin <= y_ini:
                fila_letras.append('?')
                continue
            
            celda_img = gris[y_ini:y_fin, x_ini:x_fin]
            letra, sim = reconocer_letra(celda_img, letras_referencia)
            # Normalizar letra sin tildes
            letra_normalizada = quitar_tildes(letra)
            fila_letras.append(letra_normalizada)
        
        matriz.append(fila_letras)
    
    return matriz, (img, x_coords, y_coords)


#Leer las palabras desde archivo txt
def leer_palabras(archivo_palabras):
    if not os.path.exists(archivo_palabras):
        print(f"No encontrado:  {archivo_palabras}")
        return []
    
    with open(archivo_palabras, 'r', encoding='utf-8') as f:
        contenido = f.read()

    palabras = re.findall(r'[A-ZÁÉÍÓÚÑÜ]+', contenido.upper())
    return palabras


def quitar_tildes(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn')


# uscar palabra en todas las direcciones
def buscar_palabra_en_matriz(matriz, palabra):
    if not matriz or not palabra:
        return []
    
    filas = len(matriz)
    columnas = len(matriz[0]) if filas > 0 else 0
    palabra_original = palabra.upper()
    palabra = quitar_tildes(palabra_original)  # Buscar sin tildes
    longitud = len(palabra)
    
    direcciones = [
        (0, 1),   # →
        (0, -1),  # ←
        (1, 0),   # ↓
        (-1, 0),  # ↑
        (1, 1),   # ↘
        (-1, -1), # ↖
        (1, -1),  # ↙
        (-1, 1)   # ↗
]
    
    resultados = []
    
    for i in range(filas):
        for j in range(columnas):
            for df, dc in direcciones:
                #Comprobar si la palabra cabe en esta dirección
                f_final = i + (longitud - 1) * df
                c_final = j + (longitud - 1) * dc
                
                if f_final < 0 or f_final >= filas or c_final < 0 or c_final >= columnas:
                    continue
                
                #Extraer la palabra en esta dirección
                palabra_encontrada = ""
                for k in range(longitud):
                    f = i + k * df
                    c = j + k * dc
                    palabra_encontrada += matriz[f][c]
                
                #Comprobar si coincide
                if palabra_encontrada == palabra:
                    resultados.append((i, j, (df, dc), palabra_encontrada))
    
    return resultados

def resolver_sopa_letras(imagen_path, archivo_palabras, carpeta_letras='letras'):
    
    letras_ref = cargar_letras_referencia(carpeta_letras)
    matriz, info_visual = extraer_matriz_letras(imagen_path, letras_ref)
    palabras = leer_palabras(archivo_palabras)

    resultados_totales = {}
    palabras_encontradas = []
    palabras_no_encontradas = []
    
    for palabra in palabras:
        resultados = buscar_palabra_en_matriz(matriz, palabra)
        resultados_totales[palabra] = resultados
        
        if resultados:
            palabras_encontradas.append(palabra)
        else:
            palabras_no_encontradas.append(palabra)
    
    print(f"Palabras encontradas: {len(palabras_encontradas)}/{len(palabras)}")
    print(f"Palabras no encontradas: {len(palabras_no_encontradas)}")
    
    if palabras_no_encontradas:
        print("\nPalabras no encontradas:")
        for p in palabras_no_encontradas[:10]: 
            print(f"  - {p}")
        if len(palabras_no_encontradas) > 10:
            print(f"  ... y {len(palabras_no_encontradas) - 10} más")
    
    #Visualización de resultados
    if info_visual:
        img, x_coords, y_coords = info_visual
        resultado = img.copy()
        
        #Dibujar cuadrícula
        for x in x_coords:
            cv2.line(resultado, (x, y_coords[0]), (x, y_coords[-1]), (200, 200, 200), 1)
        for y in y_coords:
            cv2.line(resultado, (x_coords[0], y), (x_coords[-1], y), (200, 200, 200), 1)
        
        #Marcar palabras encontradas con colores
        colores = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                   (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0)]
        
        color_idx = 0
        for palabra in palabras_encontradas[:200]:  
            resultados = resultados_totales[palabra]
            color = colores[color_idx % len(colores)]
            
            for fila_ini, col_ini, (df, dc), _ in resultados:
                longitud = len(palabra)
                #Dibujar línea sobre la palabra encontrada
                for k in range(longitud):
                    f = fila_ini + k * df
                    c = col_ini + k * dc
                    
                    #Encontrar coordenadas de píxeles de esta celda
                    if f < len(y_coords) - 1 and c < len(x_coords) - 1:
                        x = x_coords[c]
                        y = y_coords[f]
                        w = x_coords[c + 1] - x
                        h = y_coords[f + 1] - y
                        
                        cv2.rectangle(resultado, (x+2, y+2), (x+w-2, y+h-2), color, 2)
            
            color_idx += 1
        
        cv2.imshow('SOPA DE LETRAS RESUELTA', resultado)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return resultados_totales, matriz

if __name__ == "__main__":
    imagen = r'imagenes\2.png'
    palabras_archivo = r'palabras\2.txt'
    resolver_sopa_letras(imagen, palabras_archivo)
