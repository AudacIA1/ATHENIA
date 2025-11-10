/* eslint-disable @typescript-eslint/no-explicit-any */
import { CommonModule } from '@angular/common';
import { Component, inject, OnDestroy } from '@angular/core';
import { AlertService } from '@shared/components/alert/alert.service';
import { LucideAngularModule } from 'lucide-angular';
import { Subject, takeUntil } from 'rxjs';
import { AtheniaService, AtheniaQueryRequest } from 'src/app/services/api/athenia.service';

@Component({
  selector: 'app-athenia-voice',
  imports: [CommonModule, LucideAngularModule],
  templateUrl: './athenia-voice.html',
  styleUrl: './athenia-voice.css',
})
export class AtheniaVoice implements OnDestroy {
  private atheniaService = inject(AtheniaService);
  private alertService = inject(AlertService);
  private destroy$ = new Subject<void>();

  // Estado
  private isListening = false;
  private isSpeaking = false;
  private transcript = '';
  private response = '';

  // Reconocimiento de voz
  private recognition: any;
  private synthesis: SpeechSynthesis;
  private voices: SpeechSynthesisVoice[] = [];
  private spanishVoice: SpeechSynthesisVoice | null = null;

  // Ventana popup
  private popupWindow: Window | null = null;
  private updateInterval: any;

  constructor() {
    this.synthesis = window.speechSynthesis;
    this.loadVoices();

    if (this.synthesis.onvoiceschanged !== undefined) {
      this.synthesis.onvoiceschanged = () => this.loadVoices();
    }

    // Inicializar reconocimiento de voz
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition =
        (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
      this.recognition = new SpeechRecognition();
      this.recognition.lang = 'es-ES';
      this.recognition.continuous = false;
      this.recognition.interimResults = false;

      this.recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        this.transcript = transcript;
        this.isListening = false;
        this.updatePopupState();
        this.processVoiceQuery(transcript);
      };

      this.recognition.onerror = (event: any) => {
        console.error('Error en reconocimiento:', event.error);
        this.isListening = false;
        this.updatePopupState();
      };

      this.recognition.onend = () => {
        this.isListening = false;
        this.updatePopupState();
      };
    }

    // Escuchar mensajes de la ventana popup
    window.addEventListener('message', (event) => {
      if (event.source === this.popupWindow) {
        this.handlePopupMessage(event.data);
      }
    });
  }

  ngOnDestroy(): void {
    this.closePopup();
    this.stopSpeaking();
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }
    this.destroy$.next();
    this.destroy$.complete();
  }

  /**
   * Cargar voces en español
   **/
  private loadVoices(): void {
    this.voices = this.synthesis.getVoices();

    // Buscar voces específicas por nombre primero
    const specificVoiceNames = [
      'Microsoft Margarita Online (Natural) - Spanish (Panama)',
      'Microsoft Margarita - Spanish (Panama)',
      'Margarita',
      'es-PA-MargaritaNeural',
      'Microsoft Monica Online (Natural) - Spanish (Spain)',
      'Microsoft Monica - Spanish (Spain)',
      'Google español',
      'Spanish Female',
    ];

    // Intentar encontrar por nombre exacto
    for (const voiceName of specificVoiceNames) {
      const voice = this.voices.find((v) => v.name.toLowerCase().includes(voiceName.toLowerCase()));

      if (voice) {
        this.spanishVoice = voice;
        return;
      }
    }

    // Buscar por locale y características
    const localePreferences = [
      {
        locale: 'es-PA',
        filter: (v: SpeechSynthesisVoice) =>
          v.name.toLowerCase().includes('female') ||
          v.name.toLowerCase().includes('margarita') ||
          v.name.toLowerCase().includes('mujer'),
      },
      {
        locale: 'es-ES',
        filter: (v: SpeechSynthesisVoice) =>
          v.name.toLowerCase().includes('female') ||
          v.name.toLowerCase().includes('monica') ||
          !v.name.toLowerCase().includes('male'),
      },
      {
        locale: 'es-MX',
        filter: (v: SpeechSynthesisVoice) =>
          v.name.toLowerCase().includes('female') ||
          (!v.name.toLowerCase().includes('raul') && !v.name.toLowerCase().includes('male')),
      },
      {
        locale: 'es',
        filter: (v: SpeechSynthesisVoice) =>
          v.name.toLowerCase().includes('female') || !v.name.toLowerCase().includes('male'),
      },
    ];

    // Intentar encontrar con filtros
    for (const pref of localePreferences) {
      const voice = this.voices.find(
        (v) => v.lang.toLowerCase().startsWith(pref.locale.toLowerCase()) && pref.filter(v)
      );

      if (voice) {
        this.spanishVoice = voice;
        return;
      }
    }

    // Fallback: Cualquier voz en español que NO sea masculina
    this.spanishVoice =
      this.voices.find(
        (v) =>
          v.lang.startsWith('es') &&
          !v.name.toLowerCase().includes('raul') &&
          !v.name.toLowerCase().includes('male') &&
          !v.name.toLowerCase().includes('diego')
      ) ||
      this.voices.find((v) => v.lang.startsWith('es')) ||
      null;
  }

  /**
   * Abrir ventana popup flotante
   **/
  toggle(): void {
    if (this.popupWindow && !this.popupWindow.closed) {
      this.popupWindow.focus();
      return;
    }

    // Configuración de ventana
    const width = 450;
    const height = 700;
    const left = screen.width - width - 50;
    const top = 100;

    const features = [
      `width=${width}`,
      `height=${height}`,
      `left=${left}`,
      `top=${top}`,
      'resizable=yes',
      'scrollbars=no',
      'toolbar=no',
      'menubar=no',
      'location=no',
      'status=no',
      'directories=no',
    ].join(',');

    this.popupWindow = window.open('', 'AtheniaVoice', features);

    if (this.popupWindow) {
      this.setupPopupContent();
      this.startStateSync();
    } else {
      this.alertService.error('Popup bloqueado', 'Por favor permite popups para ATHENIA Voice');
    }
  }

  /**
   * Configurar contenido HTML de la ventana popup
   **/
  private setupPopupContent(): void {
    if (!this.popupWindow) return;

    const doc = this.popupWindow.document;

    doc.write(`
    <!DOCTYPE html>
<html lang="es">

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ATHENIA</title>
  <meta name="google" content="notranslate">
  <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🎙️</text></svg>">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #070025;
      color: #1f2937;
      min-height: 100vh;
      padding: 24px;
      display: flex;
      justify-content: center;
      align-items: center;
      overflow-x: hidden;
    }

    .container {
      border-radius: 24px;
      padding: 32px;
      width: 460px;
      animation: fadeIn 0.4s ease;
      position: relative;
    }

    .container::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 4px;
      background-size: 200% 100%;
      animation: gradientShift 3s ease infinite;
    }

    @keyframes gradientShift {
      0%, 100% {
        background-position: 0% 50%;
      }
      50% {
        background-position: 100% 50%;
      }
    }

    @keyframes fadeIn {
      from {
        opacity: 0;
        transform: translateY(20px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    .header {
      text-align: center;
      margin-bottom: 24px;
    }

    .header h1 {
      font-size: 28px;
      font-weight: 700;
      background: linear-gradient(135deg, #10b981, #6366f1);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 8px;
    }

    .status {
      font-size: 14px;
      font-weight: 500;
      color: #6b7280;
      transition: all 0.3s ease;
    }

    .status.listening {
      color: #dc2626;
      font-weight: 600;
    }

    .status.speaking {
      color: #f59e0b;
      font-weight: 600;
    }

    /* === Visualizador con neumorphism === */
    .visualizer {
      height: 240px;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 20px 0 28px;
      position: relative;
    }

    /* Esfera principal con diseño neumorphism */
    .sphere {
      width: 160px;
      height: 160px;
      border-radius: 50%;
      position: relative;
      transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Estado INACTIVO – Neumorphism azul elegante */
    .sphere.idle {
      background: #1e293b;
      box-shadow:
        12px 12px 24px rgba(0, 0, 0, 0.5),
        -12px -12px 24px rgba(45, 60, 85, 0.4),
        inset 4px 4px 8px rgba(0, 0, 0, 0.3),
        inset -4px -4px 8px rgba(60, 80, 110, 0.3),
        0 0 80px rgba(100, 180, 255, 0.3);
      animation: gentlePulse 4s ease-in-out infinite;
    }

    @keyframes gentlePulse {
      0%, 100% {
        transform: scale(1);
        box-shadow:
          12px 12px 24px rgba(0, 0, 0, 0.5),
          -12px -12px 24px rgba(45, 60, 85, 0.4),
          inset 4px 4px 8px rgba(0, 0, 0, 0.3),
          inset -4px -4px 8px rgba(60, 80, 110, 0.3),
          0 0 80px rgba(100, 180, 255, 0.3);
      }
      50% {
        transform: scale(1.03);
        box-shadow:
          14px 14px 28px rgba(0, 0, 0, 0.5),
          -14px -14px 28px rgba(45, 60, 85, 0.4),
          inset 4px 4px 8px rgba(0, 0, 0, 0.3),
          inset -4px -4px 8px rgba(60, 80, 110, 0.3),
          0 0 100px rgba(100, 180, 255, 0.4);
      }
    }

    /* Estado ESCUCHANDO – Neumorphism morado brillante */
    .sphere.listening {
      background: #1e293b;
      box-shadow:
        14px 14px 28px rgba(0, 0, 0, 0.6),
        -14px -14px 28px rgba(50, 40, 70, 0.5),
        inset 6px 6px 12px rgba(0, 0, 0, 0.4),
        inset -6px -6px 12px rgba(120, 60, 200, 0.4),
        0 0 100px rgba(160, 70, 255, 0.7),
        0 0 150px rgba(125, 45, 255, 0.5);
      animation: activeFlicker 1.5s ease-in-out infinite;
      transform: scale(1.15);
    }

    @keyframes activeFlicker {
      0%, 100% {
        filter: brightness(1.1) saturate(1.3);
      }
      50% {
        filter: brightness(1.3) saturate(1.5);
      }
    }

    /* Estado HABLANDO – Neumorphism verde esmeralda */
    .sphere.speaking {
      background: #1e293b;
      box-shadow:
        14px 14px 28px rgba(0, 0, 0, 0.6),
        -14px -14px 28px rgba(30, 60, 55, 0.5),
        inset 6px 6px 12px rgba(0, 0, 0, 0.4),
        inset -6px -6px 12px rgba(60, 180, 140, 0.4),
        0 0 100px rgba(45, 200, 140, 0.8),
        0 0 160px rgba(25, 160, 120, 0.6);
      animation: speakingGlow 0.8s ease-in-out infinite;
      transform: scale(1.1);
    }

    @keyframes speakingGlow {
      0%, 100% {
        filter: brightness(1.15) saturate(1.4);
      }
      50% {
        filter: brightness(1.35) saturate(1.6);
      }
    }

    /* Anillos orbitales sin contenedor */
    .orbit-ring {
      position: absolute;
      border: 1.5px solid rgba(100, 149, 237, 0.2);
      border-radius: 50%;
      pointer-events: none;
      top: 50%;
      left: 50%;
    }

    .orbit-ring:nth-child(1) {
      width: 200px;
      height: 200px;
      margin-left: -100px;
      margin-top: -100px;
      animation: rotateOrbit 8s linear infinite;
    }

    .orbit-ring:nth-child(2) {
      width: 240px;
      height: 240px;
      margin-left: -120px;
      margin-top: -120px;
      animation: rotateOrbit 12s linear infinite reverse;
      opacity: 0.6;
    }

    @keyframes rotateOrbit {
      from {
        transform: rotate(0deg);
      }
      to {
        transform: rotate(360deg);
      }
    }

    /* Pequeños orbes orbitales */
    .small-orb {
      position: absolute;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: radial-gradient(circle at 30% 30%,
          rgba(255, 255, 255, 0.95),
          rgba(100, 149, 237, 0.7));
      box-shadow: 0 0 15px rgba(100, 149, 237, 0.8);
      top: 50%;
      left: -5px;
      margin-top: -5px;
    }

    /* Mensajes */
    .messages-container {
      max-height: 300px;
      overflow-y: auto;
      margin-bottom: 24px;
      padding-right: 8px;
    }

    .messages-container::-webkit-scrollbar {
      width: 6px;
    }

    .messages-container::-webkit-scrollbar-track {
      background: #f3f4f6;
      border-radius: 10px;
    }

    .messages-container::-webkit-scrollbar-thumb {
      background: #d1d5db;
      border-radius: 10px;
    }

    .message-box {
      padding: 14px 18px;
      border-radius: 14px;
      margin-bottom: 16px;
      font-size: 14px;
      border: 1px solid #e5e7eb;
      transition: transform 0.2s ease;
    }

    .message-box:hover {
      transform: scale(1.02);
    }

    .transcript-box {
      background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
      border-color: #93c5fd;
    }

    .response-box {
      background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
      border-color: #86efac;
    }

    .label {
      font-weight: 700;
      margin-bottom: 8px;
      font-size: 11px;
      text-transform: uppercase;
      color: #374151;
    }

    /* Controles */
    .controls {
      display: flex;
      justify-content: center;
      gap: 16px;
      padding-top: 16px;
    }

    button {
      width: 68px;
      height: 68px;
      border: none;
      border-radius: 50%;
      font-size: 28px;
      cursor: pointer;
      color: white;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.3s ease;
      box-shadow: 0 6px 18px rgba(0, 0, 0, 0.15);
    }

    .btn-mic {
      background: #141d30;
    }

    .btn-mic.active {
      background: #141d30;
      transform: scale(1.1);
      box-shadow: 0 0 30px rgba(220, 38, 38, 0.6);
    }

    .btn-stop {
      background: #141d30;
    }

    .btn-clear {
      background: #141d30;
    }

    button:hover {
      transform: scale(1.05);
      opacity: 0.9;
    }

    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    /* Ocultar marcador flotante de Google */
    iframe.goog-te-banner-frame,
    .goog-te-banner-frame.skiptranslate,
    .goog-te-gadget-icon,
    .goog-tooltip,
    .goog-te-balloon-frame,
    .goog-te-menu-frame,
    .VIpgJd-ZVi9od-ORHb-OEVmcd {
      display: none !important;
    }

    body {
      top: 0 !important;
    }
  </style>
</head>

<body>
  <div class="container">
    <div class="header">
      <div class="status" id="status"></div>
    </div>

    <div class="visualizer" id="visualizer">
      <div class="orbit-ring">
        <div class="small-orb"></div>
      </div>
      <div class="orbit-ring">
        <div class="small-orb"></div>
      </div>
      <div class="sphere idle" id="sphere"></div>
    </div>

    <div class="messages-container" id="messages"></div>

    <div class="controls">
      <button class="btn-mic" id="btnMic" title="Iniciar/Detener reconocimiento">🎙️</button>
      <button class="btn-stop" id="btnStop" style="display:none;" title="Detener voz">⏹️</button>
      <button class="btn-clear" id="btnClear" title="Limpiar conversación">🗑️</button>
    </div>
  </div>

  <script>
    let currentTranscript = '';
    let currentResponse = '';
    const sendToParent = (type, data = {}) => window.opener.postMessage({
      type,
      ...data
    }, '*');
    const btnMic = document.getElementById('btnMic');
    const btnStop = document.getElementById('btnStop');
    const btnClear = document.getElementById('btnClear');
    const status = document.getElementById('status');
    const sphere = document.getElementById('sphere');
    const messagesDiv = document.getElementById('messages');
    btnMic.addEventListener('click', () => sendToParent('TOGGLE_LISTENING'));
    btnStop.addEventListener('click', () => sendToParent('STOP_SPEAKING'));
    btnClear.addEventListener('click', () => sendToParent('CLEAR'));
    window.addEventListener('message', (event) => {
      const {
        type,
        isListening,
        isSpeaking,
        transcript,
        response
      } = event.data;
      if (type !== 'UPDATE_STATE') return;
      if (isListening) {
        status.textContent = 'Escuchando...';
        status.className = 'status listening';
        sphere.className = 'sphere listening';
        btnMic.classList.add('active');
      } else if (isSpeaking) {
        status.textContent = 'Hablando...';
        status.className = 'status speaking';
        sphere.className = 'sphere speaking';
        btnStop.style.display = 'block';
        btnMic.disabled = true;
      } else {
        status.textContent = '';
        status.className = 'status';
        sphere.className = 'sphere idle';
        btnMic.classList.remove('active');
        btnMic.disabled = false;
        btnStop.style.display = 'none';
      }
      if (transcript !== currentTranscript || response !== currentResponse) {
        currentTranscript = transcript;
        currentResponse = response;
        messagesDiv.innerHTML = '';
        if (transcript) {
          messagesDiv.innerHTML += \`
            <div class="message-box transcript-box">
              <div class="label">👤 Tú dijiste:</div>
              <div>\${transcript}</div>
            </div>
          \`;
        }
        if (response) {
          messagesDiv.innerHTML += \`
            <div class="message-box response-box">
              <div class="label">🤖 ATHENIA responde:</div>
              <div>\${response}</div>
            </div>
          \`;
        }
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
      }
    });

    document.addEventListener('keydown', (e) => e.key === 'Escape' && window.close());
  </script>
</body>

</html>
  `);

    doc.close();
  }

  /**
   * Sincronizar estado con popup cada 100ms
   */
  private startStateSync(): void {
    this.updatePopupState();
    this.updateInterval = setInterval(() => {
      if (this.popupWindow && !this.popupWindow.closed) {
        this.updatePopupState();
      } else {
        clearInterval(this.updateInterval);
        this.popupWindow = null;
      }
    }, 100);
  }

  /**
   * Enviar estado actual al popup
   */
  private updatePopupState(): void {
    if (this.popupWindow && !this.popupWindow.closed) {
      this.popupWindow.postMessage(
        {
          type: 'UPDATE_STATE',
          isListening: this.isListening,
          isSpeaking: this.isSpeaking,
          transcript: this.transcript,
          response: this.response,
        },
        '*'
      );
    }
  }

  /**
   * Manejar mensajes del popup
   **/
  private handlePopupMessage(data: any): void {
    switch (data.type) {
      case 'TOGGLE_LISTENING':
        if (this.isListening) {
          this.stopListening();
        } else {
          this.startListening();
        }
        break;

      case 'STOP_SPEAKING':
        this.stopSpeaking();
        break;

      case 'CLEAR':
        this.clearConversation();
        break;
    }
  }

  /**
   * Iniciar reconocimiento de voz
   **/
  private startListening(): void {
    if (!this.recognition) {
      alert('Tu navegador no soporta reconocimiento de voz');
      return;
    }

    this.transcript = '';
    this.response = '';
    this.isListening = true;
    this.updatePopupState();

    try {
      this.recognition.start();
    } catch (error) {
      console.error('Error al iniciar reconocimiento:', error);
      this.isListening = false;
      this.updatePopupState();
    }
  }

  /**
   * Detener reconocimiento
   **/
  private stopListening(): void {
    if (this.recognition && this.isListening) {
      this.recognition.stop();
      this.isListening = false;
      this.updatePopupState();
    }
  }

  /**
   * Procesar pregunta de voz
   **/
  private processVoiceQuery(question: string): void {
    const lowerQuestion = question.toLowerCase();

    if (lowerQuestion.includes('adiós') || lowerQuestion.includes('cerrar')) {
      this.speak('Hasta luego. Que tengas un buen día.');
      setTimeout(() => this.closePopup(), 3000);
      return;
    }

    if (lowerQuestion.includes('limpiar') || lowerQuestion.includes('borrar')) {
      this.clearConversation();
      this.speak('Conversación limpiada.');
      return;
    }

    if (lowerQuestion.includes('ayuda')) {
      this.speak(
        'Puedes decir: Adiós para cerrar, Limpiar para borrar, o hacer cualquier pregunta sobre tus documentos.'
      );
      return;
    }

    // Enviar a ATHENIA
    const request: AtheniaQueryRequest = {
      question: question,
      use_cache: true,
    };

    this.atheniaService
      .askQuestion(request)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          this.response = response.answer;
          this.updatePopupState();
          this.speak(response.answer);
        },
        error: (error) => {
          console.error('Error:', error);
          const errorMsg = 'Lo siento, hubo un error. Intenta de nuevo.';
          this.response = errorMsg;
          this.updatePopupState();
          this.speak(errorMsg);
        },
      });
  }

  /**
   * Hablar texto
   **/
  private speak(text: string): void {
    this.stopSpeaking();

    if (this.voices.length === 0) {
      this.loadVoices();
    }

    // Dividir texto en chunks si es muy largo (más de 200 caracteres)
    const maxLength = 200;
    if (text.length > maxLength) {
      this.speakLongText(text);
      return;
    }

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'es-ES';
    utterance.rate = 1.1;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    if (this.spanishVoice) {
      utterance.voice = this.spanishVoice;
    }

    utterance.onstart = () => {
      this.isSpeaking = true;
      this.updatePopupState();
    };

    utterance.onend = () => {
      this.isSpeaking = false;
      this.updatePopupState();
    };

    utterance.onerror = (event) => {
      console.error('Error en síntesis de voz:', event);
      this.isSpeaking = false;
      this.updatePopupState();
    };

    // Delay para sincronización con UI
    setTimeout(() => {
      this.synthesis.speak(utterance);
    }, 100);
  }

  /**
   * Hablar textos largos dividiéndolos en partes
   **/
  private speakLongText(text: string): void {
    const sentences = text.match(/[^/.!/?]+[/.!/?]+/g) || [text];
    let index = 0;

    const speakNext = () => {
      if (index >= sentences.length) {
        this.isSpeaking = false;
        this.updatePopupState();
        return;
      }

      const utterance = new SpeechSynthesisUtterance(sentences[index].trim());
      utterance.lang = 'es-ES';
      utterance.rate = 1.1;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;

      if (this.spanishVoice) {
        utterance.voice = this.spanishVoice;
      }

      utterance.onend = () => {
        index++;
        speakNext();
      };

      utterance.onerror = () => {
        this.isSpeaking = false;
        this.updatePopupState();
      };

      if (index === 0) {
        this.isSpeaking = true;
        this.updatePopupState();
      }

      // Pequeño delay entre oraciones
      setTimeout(() => {
        this.synthesis.speak(utterance);
      }, 50);
    };

    speakNext();
  }

  /**
   * Detener habla
   **/
  private stopSpeaking(): void {
    if (this.synthesis.speaking) {
      this.synthesis.cancel();
      this.isSpeaking = false;
      this.updatePopupState();
    }
  }

  /**
   * Limpiar conversación
   */
  private clearConversation(): void {
    this.transcript = '';
    this.response = '';
    this.stopSpeaking();
    this.updatePopupState();
  }

  /**
   * Cerrar popup
   */
  private closePopup(): void {
    if (this.popupWindow && !this.popupWindow.closed) {
      this.popupWindow.close();
    }
    this.popupWindow = null;
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }
  }
}
