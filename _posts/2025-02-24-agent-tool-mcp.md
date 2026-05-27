---
date: 2025-02-24
title: Agentic AI Kavramları

image: /assets/img/ss/2025-02-24-agents-tools-mcp/agentic-ai-cover.jpg
#categories: [nlp]
tags: [machine-learning,agentic-ai,agent,mcp]
mermaid: true
published: true
math: true
description: Yapay zeka dünyası, sadece soru sorduğumuz chatbot döneminden çıkıp kendi içinde organize olan dijital organizasyonlara dönüşüyor. Bu yazıda Agentic AI, AI Agents, Agent to Agent, Human-in-the-Loop, Guardrails, Tool ve MCP gibi günümüzün en çok karıştırılan kavramlarını modern bir şirket hiyerarşisi üzerinden ele alıyoruz. Karmaşık mimarileri ve terzi usulü entegrasyon süreçlerini bir kenara bırakıp, bu yeni otonom ekosistemin arka planda nasıl çalıştığını en yalın haliyle masaya yatırıyoruz.
---

## Chatbotlardan Sonrası: Agent, Tool ve MCP Kavramları

Son zamanlarda nereye baksak "Agentic AI", "AI Agents", "MCP" gibi kısaltmalar havada uçuşuyor. Sektör o kadar hızlı tüketiyor ki kavramları, bazen takip etmek gerçekten yorucu olabiliyor. Prompt yazıp cevap aldığımız o klasik chatbot dönemini yavaş yavaş geride bırakıyoruz; artık yapay zekanın arka planda gerçekten iş yapmasını, kararlar almasını beklediğimiz bir evredeyiz.

Ama bu kavramlar birbirine çok karışabiliyor. "Ajan dediğin şey zaten kod değil mi?", "Tool ile ajan arasındaki sınır nerede başlıyor?", "Anthropic'in çıkardığı şu MCP aslında ne işe yarıyor?" gibi soruları bir süredir ben de kendi içimde karıştırmadan netleştirmeye çalışıyordum.

Kafadaki o spagetti görüntüyü çözmek için en güzel yöntem, bunları modern, dikey hiyerarşisi olan bir teknoloji şirketinin çalışma modeline benzeterek somutlaştırmak. Gelin terimleri bir kenara bırakıp mutfağa girelim.

![alt text](/assets/img/ss/2025-02-24-agents-tools-mcp/agent-tool-mcp.jpg) 

## Sorun: Her şeyi ben bilirim

İlk başlarda bir LLM'e (Büyük Dil Modeli) her şeyi yaptırmaya çalışıyorduk. Hem kod yazsın, hem bütçe analizi yapsın, hem gitsin mailleri kontrol etsin... Bu, bir şirkette tek bir kişiyi işe alıp ona hem CEO'luk, hem muhasebecilik, hem de yazılımcılık yaptırmaya benziyor. Günün sonunda o çalışan nasıl tükenirse, yapay zeka da saçmalamaya (hallucination) başlıyor.

İşte bu yüzden işi mikro parçalara bölmeye başladık. Buna genel olarak `Agentic AI` (yani sistemlerin artık daha otonom, kendi kararlarını verebilen bir felsefeyle tasarlanması) diyoruz. Bu felsefenin içindeki her bir `rol` ise birer `AI Agent (Ajan)`.

## Şirket İçi Hiyerarşi: Ajanları Yöneten Ajanlar

![alt text](/assets/img/ss/2025-02-24-agents-tools-mcp/agentic-company.jpg)

Gerçek bir ajan mimarisi kurduğunuzda, aslında içeride mini bir şirket kurmuş oluyorsunuz. Tek bir ajan her işe koşmuyor; aksine içeride tıpkı kurumsal dünyadaki gibi dikey bir ast-üst ilişkisi dönüyor. İşi en küçük parçacıklarına kadar bölüp, birbirini denetleyen bir bütün oluşturabiliyoruz.

Diyelim ki sisteme tek bir cümlelik görev verdiniz: "Yeni ürünün lansman kampanyasını kurgula ve bütçesini optimize et."

Arka planda süreç, tahtada çizdiğimiz hiyerarşik akışla adım adım aşağıya doğru kırılıyor:

  - Yönetim Masası (C-Level / CEO Agent): Gelen bu büyük ve stratejik hedefi tek başına yapmaya çalışmaz. Büyük resmi analiz ederek stratejik planlamayı yapar ve ana görevi alt parçalara böler.

  - Direktörler: CEO'dan gelen stratejik planı Operasyon Direktörü ve Teknoloji Direktörü ajanları devralır. Görevleri, bu büyük hedefi kendi sorumluluk alanlarına göre daha somut alt hedeflere dönüştürmek ve bölmektir.

  - Müdürler: Direktörlerden gelen kırılımlar Pazarlama Müdürü ve Finans Müdürü ajanlarının önüne düşer. Müdürlerin işi koordinasyon ve kalite kontroldür. Alt ekibin iş dağılımını yapar ve gelen çıktıları yukarıya raporlamadan önce denetlerler.

  - İşçi Agentlar: En alt katmanda ise sadece kendi uzmanlığına odaklanmış mikro çalışanlar yer alır. Pazarlama müdürüne bağlı çalışan Metin Yazarı ve Grafik Tasarım ajanları kreatif içerikleri üretirken; Finans müdürüne bağlı çalışan Gider Analisti ajanı bütçe hesaplamalarına odaklanır.

Metin yazarı ve grafik tasarımcı ajanlar işi bitirip pazarlama müdürüne sunar. Finans tarafındaki analist ajan ise veri akışını sağlamak için MCP köprüsünü kullanarak harici Tools (Araçlar) katmanına ulaşır; SQL veritabanından, Excel'den verileri çeker ve bütçeyi çıkartır.

Tıpkı gerçek hayattaki gibi, hiç kimse birbirinin alanına girmeden, mikro görevler hiyerarşik olarak yukarıya doğru birleşir ve günün sonunda yönetim masasına kusursuz bir "Başarı Analizi" olarak teslim edilir.

## Peki Bu Çalışanların Önündeki/Kullandığı Araçlar veya Ekranlar Ne? (Tools)

En alt katmandaki o Uzman Ajanların iş üretebilmesi için dış dünya ile etkileşime girmesi gerekir. Önüne bilgisayar koymadığınız bir finans analisti ne kadar zeki olursa olsun iş yapamaz. İşte ajanın internette arama yapmasını, şirketin SQL veritabanına girmesini, Slack'ten mesaj atmasını ya da bir hesaplama yapmasını sağlayan o harici fonksiyonlara `Tool (Araç)` diyoruz. 

Buradaki en kritik ayrım şu: **Tool'un bir iradesi yoktur.** Bir SQL veritabanı kendi kendine çalışıp analiz yapmaz. Ama bizim Gider Analisti Ajanı, müdüründen aldığı talimat doğrultusunda "Şu an geçmiş dataya bakmam lazım" der, SQL veritabanı aracını (Tool) çağırır, gelen ham veriyi işler ve anlamlı bir rapor haline getirip üstüne teslim eder. yani ajan stratejiyi belirler, tool ise sadece tetiklenen bir alettir.

## MCP (Model Context Protocol) Bu Hikayenin Neresinde?

Geldik son zamanların en popüler ama biraz zor anlaşılan kavramına. Şirkette onlarca ajan var ve alt tarafta da kullanılması gereken yüzlerce yazılım, API ve veritabanı (PostgreSQL, Notion, GitHub, Salesforce vb.) mevcut. Eskiden her bir ajanın, her bir veritabanına erişebilmesi için yazılımcıların her seferinde özel, terzi usulü entegrasyon kodları yazması gerekiyordu. Tam bir spagetti karmaşası. Anthropic'in duyurduğu MCP, aslında bu şirket içindeki ortak "Type-C Priz Standardı" ya da güvenli şirket içi network ağı. MCP sayesinde, şirkete bağlamak istediğiniz herhangi bir veritabanını veya harici aracı (Tool) bir kez bu standarda uygun hale getiriyorsunuz. Sonrasında ister en alttaki stajyer ajan olsun, ister en üstteki CEO ajan, hepsi o veriye "tak-çalıştır" şeklinde pürüzsüzce erişebiliyor. Altyapı karmaşasını ortadan kaldıran, işin tamamen mimari ve standardizasyon kısmı yani.

## Şirket Sınırlarının Dışına Çıkmak: Agent-to-Agent İletişimi

Buraya kadar kurduğumuz hiyerarşide, tüm ajanlar aynı şirketin (yani aynı sistemin veya framework'ün) içindeydi. Peki ya bizim finans müdürü ajanı, şirket dışındaki bir bankanın yapay zeka ajanıyla doğrudan konuşup anlaşabilseydi? ya da pazarlama ajanı, harici bir reklam ajansının yapay zekasına direkt iş paslayabilseydi?

İşte Google ve sektör devlerinin şu an standardını oturtmaya çalıştığı vizyon tam olarak bu: Agent-to-Agent (A2A) dünyası.

Bunu da şirket metaforumuz üzerinden düşünelim:

Bugün şirketler birbirleriyle nasıl iş yapıyor? Mailler atılıyor, toplantılar organize ediliyor, sözleşmeler imzalanıyor... İki farklı şirketin insan çalışanları günlerce iletişim kurmaya çalışıyor. Agent-to-Agent protokolleri oturduğunda ise süreç şuna dönecek:

  - Bizim şirketteki Satın Alma Ajanı, ofise yeni bilgisayarlar alınması gerektiğine karar verecek.

  - Telefonu kaldırıp insanları aramak yerine, doğrudan teknoloji tedarikçisi olan X şirketinin Satış Ajanı ile dijital ortamda bir araya gelecek.

  - İki harici ajan kendi aralarında saniyeler içinde pazarlık yapacak, stok durumunu kontrol edecek, fiyat tekliflerini pürüzsüz bir protokolle netleştirip faturayı kesecek.

Yani yapay zeka artık sadece kendi içindeki alt departmanları yönetmekle kalmayacak; şirket dışındaki diğer bağımsız yapay zekalarla da ortak bir dilde "pazarlık yapabilen, iş birliği kurabilen" birer dijital ticari aktöre dönüşecek.

İşte o zaman chatbot dünyasından tamamen sıyrılıp, ajanların birbiriyle ticaret yaptığı, veri takas ettiği ve operasyon yürüttüğü gerçek bir Ajan Ekonomisi (Agent Economy) evresine geçmiş olacağız.

## Yapay Zeka Şirketinin Güvenlik ve Onay Duvarları: HITL ve Guardrails

Hiyerarşiyi kurduk, departmanları böldük, hatta ajanların dış dünyadaki diğer ajanlarla konuşabileceği (A2A) bir gelecekten bahsettik. Fakat burada çok kritik bir soru işareti doğuyor: **Biz bu ajanlara gerçekten ne kadar güvenebiliriz?**

Bir sabah uyandığımızda, Finans Müdürü ajanın kendi kafasına göre 1 milyon dolarlık bir harcamayı onaylamadığından ya da Metin Yazarı ajanın dışarıya şirket sırlarını sızdırmadığından nasıl emin olacağız? İşte tam bu noktada, üretken yapay zekayı "başıboş bir yazılım" olmaktan çıkarıp kurumsal bir kalıba sokan iki kavram devreye giriyor:`HITL (Human-in-the-Loop)` ve `Guardrails`.


### Human-in-the-Loop (HITL): Son Söz Her Zaman İnsanın

Türkçeye "Sürece İnsanın Dahil Olması" olarak çevirebileceğimiz HITL, otonom sistemlerin arasına koyduğumuz akıllı duraklama noktalarıdır. Şirketteki karşılığı tam olarak "Yönetim Kurulu Onayı" veya "Islak İmza" mekanizmasıdır.

Ajanlar ne kadar zeki ve otonom olursa olsun, riskli adımlarda süreç durur ve bir insandan onay bekler. Örneğin; Finans Ajanı bütçeyi hesaplar, Metin Yazarı ajansı reklamı kurgular, her şey hazırdır. Ancak o reklamın canlıya alınması veya bütçenin hesaptan çıkması aşamasında sistem askıya alınır. Ajan, insana dönüp "Her şeyi hazırladım, onaylıyor musunuz?" der. İnsan "Onayla" butonuna bastığı anda süreç kaldığı yerden otonom şekilde akmaya devam eder.

Böylece kontrolü kaybetmeden, operasyonel yükü yapay zekaya devretmiş oluruz.

### Guardrails: Ajanların Kırmızı Çizgileri

Şirketlerin "İç Tüzüğü", "KVKK Politikaları" veya "Regülasyonları" neyse, yapay zekadaki Guardrails (Korkuluklar) da tam olarak odur. Ajanların hareket alanını, bütçesini, erişebileceği veri sınırlarını ve konuşma üslubunu kısıtlayan sert yazılımsal duvarlardır.

Fakat burada gözden kaçırılmaması gereken çok kritik bir nokta var: Bu korkuluklar sadece işe başlamadan önce değil, iş bittikten sonra da devreye girer. Yani hem "Giriş (Input)" hem de "Çıkış (Output)" kontrolü yapılır.

  - **İş Başlamadan Önce (Input Guardrails):** Bu, şirkete gelen taleplerin daha kapıda filtrelenmesidir. Örneğin; bir müşterinin ajanımıza şirketin kaynak kodlarını veya diğer müşterilerin KVKK kapsamındaki verilerini sorması durumunda, guardrail devreye girer. Ajanın bu soruyu hiç okumasına veya işlemesine izin vermeden talebi kapıda reddeder.

  - **İş Bittikten Hemen Sonra (Output Guardrails):** Bu da bir çalışanın hazırladığı raporun veya müşteriye göndereceği e-postanın, şirketten çıkmadan tam bir saniye önce son bir kez süzgeçten geçirilmesidir. Ajan arka planda harika bir analiz yapmış olabilir ama halüsinasyon görüp çıktının içine yanlış bir mali veri eklemiş ya da kurumsal üsluba uymayan agresif bir cümle kurmuş olabilir. Çıkış guardrail'ları, o e-posta müşterinin kutusuna düşmeden hemen önce devreye girer, çıktıyı tarar, eğer bir ihlal varsa maili durdurur ve ajana "Bu kurumsal politikalara uymuyor, yeniden düzenle" uyarısı yapar.

Guardrails sayesinde bir ajanın önüne şu tarz kesin kurallar koyarsınız:

  - "Müşteriyle konuşurken asla şirket içi hassas mali verileri dışarıya sızdırma." (Çıkış Kontrolü)

  - "Kullanıcı sana manipülatif bir soruyla gelirse bunu daha işleme almadan reddet." (Giriş Kontrolü)

  - "Tek seferde 50 dolardan fazla tool/API çağrısı yapma." (Maliyet Kontrolü)

Ajan arka planda ne kadar karmaşık düşünürse düşünsün, kapıdaki veya çıkıştaki bu korkuluklara çarptığı anda durdurulur. Böylece hem girdiyi hem de çıktıyı çift taraflı güvenceye alarak şirketin itibarını ve verisini korumuş oluruz.

![alt text](/assets/img/ss/2025-02-24-agents-tools-mcp/multiagent.jpg)

## Özetlemek Gerekirse


Günün sonunda yapay zeka dünyası artık sadece soru-cevap üzerine kurulu chatbotlardan ibaret değil; arka planda birbiriyle konuşan, harici araçları tetikleyen ve güvenlik filtrelerinden geçen otonom bir Ajanlar Topluluğu (Multi-Agent System) mimarisine dönüşüyor.

Bu projedeki tüm teknik taşları yerine oturtacak olursak:

  - `Agentic AI:` Sistemlerin statik cevaplar üretmek yerine, kendilerine verilen hedef doğrultusunda kendi iş akışlarını otonom planlayıp yürütebilmesi felsefesidir.

  - `AI Agent (Ajan)`: Bu topluluğun içindeki, belirli bir uzmanlığı, görevi ve kendi sınırları içinde karar verme yetkisi olan bağımsız yazılım modülleridir.

  - `Tool (Araç)`: Ajanların dış dünya ile etkileşime girmesini, veri çekmesini veya işlem yapmasını sağlayan harici fonksiyonlar ve API'lerdir (SQL veritabanı, Slack, web arama).

  - `MCP (Model Context Protocol)`: Ajanlar ile bu araçlar (Tools) arasına çekilen, entegrasyon karmaşasını bitiren evrensel "tak-çalıştır" bağlantı standardıdır.

  - `HITL & Guardrails`: Sistemin hata yapmasını veya veri sızdırmasını engelleyen; girdiyi/çıktıyı denetleyen güvenlik duvarları (Guardrails) ve kritik adımlardaki insan onay (HITL) mekanizmasıdır.

  - `Agent-to-Agent (A2A)`: Bizim kurduğumuz bu lokal ajan topluluğunun, dış dünyadaki tamamen farklı ajan topluluklarıyla ortak bir protokol üzerinden doğrudan iletişim kurabilmesi ve veri takası yapabilmesi vizyonudur.

