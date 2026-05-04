from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import random
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Simulación de modelo IA con respuestas variadas
TEJIDOS = {
    'Autopsia': [
        {'tejido': 'Tejido necrótico', 'confianza': 88.5, 'notas': 'Áreas de necrosis coagulativa. Posible infarto.'},
        {'tejido': 'Tejido conectivo fibroso', 'confianza': 91.2, 'notas': 'Fibrosis intersticial moderada.'},
        {'tejido': 'Tejido adiposo con infiltrado', 'confianza': 86.7, 'notas': 'Infiltrado inflamatorio crónico.'},
    ],
    'Biopsia': [
        {'tejido': 'Tejido epitelial estratificado', 'confianza': 89.2, 'notas': 'Sin atipias significativas. Epitelio conservado.'},
        {'tejido': 'Tejido conectivo denso', 'confianza': 94.5, 'notas': 'Fibras colágenas organizadas. Estroma normal.'},
        {'tejido': 'Tejido glandular', 'confianza': 87.8, 'notas': 'Estructuras acinares preservadas.'},
        {'tejido': 'Tejido adiposo', 'confianza': 91.8, 'notas': 'Adipocitos maduros sin alteraciones.'},
    ],
    'Citologia': [
        {'tejido': 'Células epiteliales escamosas', 'confianza': 85.3, 'notas': 'Sin cambios displásicos. Citología normal.'},
        {'tejido': 'Células inflamatorias mixtas', 'confianza': 82.7, 'notas': 'Predominio de neutrófilos. Proceso agudo.'},
        {'tejido': 'Células mesenquimales', 'confianza': 79.5, 'notas': 'Muestra adecuada. Células fusiformes.'},
    ],
}

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'ok', 'message': 'MicroHisto Backend activo', 'timestamp': datetime.now().isoformat()})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        tipo = request.form.get('tipo', 'Biopsia')
        etario = request.form.get('etario', 'Adulto')
        sexo = request.form.get('sexo', 'Desconocido')
        patologias = request.form.get('patologias', '')
        
        # Obtener imágenes
        imagenes = request.files.getlist('imagenes')
        if not imagenes:
            imagenes_keys = [k for k in request.files.keys()]
            imagenes = [request.files[k] for k in imagenes_keys]
        
        resultados = []
        tejidos_disponibles = TEJIDOS.get(tipo, TEJIDOS['Biopsia'])
        
        for i, img in enumerate(imagenes):
            # Seleccionar tejido basado en el tipo de muestra
            base = tejidos_disponibles[i % len(tejidos_disponibles)]
            
            # Ajustar confianza según datos del paciente
            confianza_base = base['confianza']
            if patologias:
                confianza_base -= random.uniform(1, 5)  # Patologías añaden incertidumbre
            
            # Para fetos, posible tejido placentario
            notas_extra = ''
            if etario == 'Feto' and random.random() > 0.5:
                notas_extra = ' Posible origen placentario. '
            
            resultados.append({
                'tejido': base['tejido'],
                'confianza': round(confianza_base, 1),
                'notas': base['notas'] + notas_extra,
                'imagen_index': i,
            })
        
        return jsonify({
            'success': True,
            'resultados': resultados,
            'metadata': {
                'tipo_muestra': tipo,
                'grupo_etario': etario,
                'sexo': sexo,
                'total_imagenes': len(imagenes),
                'modelo': 'MicroHisto IA v1.0 (simulación)',
            }
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
